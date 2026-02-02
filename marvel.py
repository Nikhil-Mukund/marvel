
'''
MARVEL: A Multi‑Agent Research Validator and Enabler using LLMs

        MARVEL is a locally deployable, open‑source framework for domain‑aware
        question answering and assisted research. It combines a fast path for
        straightforward queries with a DeepSearch mode that integrates
        retrieval‑augmented generation (RAG) and Monte‑Carlo Tree Search to
        explore complementary sub‑queries and synthesise evidence without
        duplication.  It draws on a curated semantic index of arXiv literature,
        theses, public LIGO documents and logbooks, and targeted web searches,
        with stable citation tracking throughout.  Evaluations show MARVEL
        performs comparably to commercial systems on literature‑focused questions
        and significantly better on detector‑operations queries.  The system
        ships as an auditable, lab‑hosted service and can be adapted to other
        scientific domains without sending data outside the system.

        **Developed by: Nikhil Mukund, Yifang Luo, Fan Zhang, Erik Katsavounidis and
        Lisa Barsotti, with support from the MIT Kavli Institute for Astrophysics
        and Space Research & LIGO Laboratory, and the NSF AI Institute for Artificial
        Intelligence and Fundamental Interactions.
'''

# Always check README.md for installation, setup instructions, usage and recent modifications.

# Run this code using: $ streamlit run marvel.py

# Author: Nikhi Mukund 
# Institute: MIT Kavli - LIGO Laboratory - NSF IAIFI

from __future__ import annotations
from typing import Any, Iterable, List, Tuple
import copy  # make sure copy is imported if not already


import streamlit as st
import json

if "page_config_set" not in st.session_state:
    st.set_page_config(layout="wide", initial_sidebar_state="expanded")
    st.session_state.page_config_set = True


import platform
from init import faiss_embeddings_init, ragatouille_init
from libs.faiss import save_faiss_vectorstore, load_faiss_vectorstore, get_docs_from_faiss_vectorstore
from libs.print import print_wide_line_message, pretty_print_docs,streamlit_add_msg,streamlit_add_line,streamlit_add_bold_heading
from libs.retrievers import fetch_arxiv_abstracts,generate_wiki_summary_from_langchain_docs
from libs.prompts import prompt
from libs.content import preprocess_page_content_text, diagnostic_print_RAG_chunks, print_langchain_documents

from libs.regex import escape_braces, nicely_format_latex_eqns, validate_and_replace_urls, add_color_filename_tags, convert_all_latex_delimiters,replace_plus_in_url_paths,wrap_urls_in_angle_brackets,wrap_urls_in_metadata,normalize_quotes,extract_answer_or_raw,_pp_output_to_text
from libs.retrievers import CustomRetriever_Tavily, CustomRetriever_BM25, CustomRetriever_FAISS, CustomRetriever_DuckDuckGo, enhanced_retriever_duckduckgo,enhanced_retriever_tavily, verify_and_filter_retrieved_docs_v2,get_best_VecDB_info, combine_langchain_documents,enhanced_retriever_FAISS,enhanced_retriever_BM25, heyligo_search,heyligo_selenium_search,verify_and_filter_retrieved_docs_v2_parallel,ensembele_superposition_answer_from_docs
from libs.extractors import extract_pdf_data_V3, extract_text_data, extract_latex_data, extract_JSONL_data, extract_heyligo_ProFreports_csv_data, extract_alpaca_json_data
from libs.summarizer_v3 import langgraph_summarize_documents
from libs.utilities import load_acronyms, get_acronym_definition

from langchain.memory import ConversationSummaryBufferMemory
import asyncio
from langchain.retrievers.document_compressors.chain_filter_prompt import prompt_template
from init import init_environment
import numpy as np
from langchain.memory import ConversationSummaryBufferMemory
from sklearn.cluster import KMeans
from datetime import datetime, date
import httpx, atexit
import string

from langchain_core.output_parsers import BaseOutputParser
from langchain.retrievers.document_compressors.chain_filter_prompt import prompt_template
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
import subprocess
import argparse
import time
import re
import random
from fuzzywuzzy import process
from tqdm import tqdm
from collections import Counter
import string
import copy
from langchain_groq import ChatGroq

import difflib, re
from typing import Tuple, Optional

from PIL import Image

# -----------------------------------------------------------------------------
# Reasoning trace (high-level) collection for UI debugging.
# This is NOT chain-of-thought. It records tool/agent steps and retrieval stats.
# -----------------------------------------------------------------------------
from contextlib import contextmanager

st.session_state.setdefault("reasoning_trace", [])
st.session_state.setdefault("reasoning_trace_meta", {})
st.session_state.setdefault("reasoning_trace_max_items", 120)

def trace_reset(user_query: str = "") -> None:
    """Start a fresh trace for a new user turn."""
    try:
        st.session_state.reasoning_trace = []
        st.session_state.reasoning_trace_meta = {
            "query": user_query,
            "started_at": time.time(),
        }
    except Exception:
        pass

def trace_add(title: str, detail: str = "") -> None:
    """Append a high-level trace event (safe to display to users)."""
    try:
        if "reasoning_trace" not in st.session_state or not isinstance(st.session_state.reasoning_trace, list):
            st.session_state.reasoning_trace = []
        max_items = int(st.session_state.get("reasoning_trace_max_items", 120))
        if len(st.session_state.reasoning_trace) >= max_items:
            # Add truncation note only once
            if not any(isinstance(e, dict) and e.get("title") == "Trace truncated" for e in st.session_state.reasoning_trace):
                st.session_state.reasoning_trace.append({
                    "title": "Trace truncated",
                    "detail": f"Reached max items ({max_items}). Further events omitted."
                })
            return
        st.session_state.reasoning_trace.append({
            "title": str(title).strip(),
            "detail": str(detail).strip() if detail else ""
        })
    except Exception:
        pass

@contextmanager
def traced_spinner(label: str, detail: str = ""):
    """Spinner wrapper that also records a trace event."""
    trace_add(label, detail)
    with st.spinner(label):
        yield

def format_reasoning_trace_md(trace_items: List[dict]) -> str:
    """Convert trace items into Markdown similar to ChatGPT 'thinking' style."""
    if not trace_items:
        return "_(no trace recorded)_"
    lines: List[str] = []
    for it in trace_items:
        if not isinstance(it, dict):
            continue
        title = (it.get("title") or "").strip()
        detail = (it.get("detail") or "").strip()
        if not title and not detail:
            continue
        if title:
            lines.append(f"- **{title}**")
        if detail:
            for ln in detail.splitlines():
                ln = ln.rstrip()
                if ln == "":
                    lines.append("  >")
                else:
                    lines.append(f"  > {ln}")
        lines.append("")
    return "\n".join(lines).strip()

def render_reasoning_trace_expander(trace_items: List[dict], *, label: str = "🧠 Show reasoning trace") -> None:
    if not trace_items:
        return
    with st.expander(label, expanded=False):
        meta = st.session_state.get("reasoning_trace_meta") or {}
        started_at = meta.get("started_at")
        if isinstance(started_at, (int, float)):
            elapsed = time.time() - float(started_at)
            st.caption(f"Trace events: {len(trace_items)} • elapsed: {elapsed:.2f}s")
        st.markdown(format_reasoning_trace_md(trace_items))

def attach_trace_to_last_assistant_message(trace_items: List[dict]) -> None:
    """Attach trace to the most recent assistant message in chat_history (for persistence across reruns)."""
    if not trace_items:
        return
    try:
        hist = st.session_state.get("chat_history") or []
        if not isinstance(hist, list):
            return
        for msg in reversed(hist):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                # Don't overwrite existing trace
                if "trace" not in msg:
                    msg["trace"] = copy.deepcopy(trace_items)
                break
    except Exception:
        pass

## soft kill
class SearchCancelled(Exception):
    pass
def _request_stop_search():
    st.session_state.stop_search_requested = True
def _check_stop_search():
    if st.session_state.get("stop_search_requested", False):
        st.session_state.stop_search_requested = False
        raise SearchCancelled()
## 

st.title("MARVEL : A Multi‑Agent Research Validator and Enabler")

# Place this near the top of your app or inside the sidebar
with st.sidebar:
    with st.expander("ℹ️ About MARVEL", expanded=False):
        st.markdown("""
        **MARVEL: A Multi‑Agent Research Validator and Enabler**

        MARVEL is a locally deployable, open‑source framework for domain‑aware
        question answering and assisted research. It combines a fast path for
        straightforward queries with a DeepSearch mode that integrates
        retrieval‑augmented generation (RAG) and Monte‑Carlo Tree Search to
        explore complementary sub‑queries and synthesise evidence without
        duplication.  It draws on a curated semantic index of arXiv literature,
        theses, public LIGO documents and logbooks, and targeted web searches,
        with stable citation tracking throughout.  Evaluations show MARVEL
        performs comparably to commercial systems on literature‑focused questions
        and significantly better on detector‑operations queries.  The system
        ships as an auditable, lab‑hosted service and can be adapted to other
        scientific domains without sending data outside the system.

        **Developed by:** Nikhil Mukund, Yifang Luo, Fan Zhang, Erik Katsavounidis and
        Lisa Barsotti, with support from the MIT Kavli Institute for Astrophysics
        and Space Research & LIGO Laboratory, and the NSF AI Institute for Artificial
        Intelligence and Fundamental Interactions.
        """)



from config import config
configs = config.load_config()



default_config = configs
# Initialize session state for modified configuration if not already set.
if "modified_config" not in st.session_state:
    st.session_state.modified_config = copy.deepcopy(default_config)
# Mirror modified config into a global variable if needed.
st.session_state.configs = st.session_state.modified_config




# --- Agent Settings sidebar ---
st.sidebar.header("Agent Settings")

# Ensure defaults exist in session_state for these flags
st.session_state.setdefault("tavily_selected", True)
st.session_state.setdefault("faiss_selected", True)
st.session_state.setdefault("heyligo_selected", True)
st.session_state.setdefault("bm25_selected", True)
st.session_state.setdefault("early_stop_enabled", True)
st.session_state.setdefault("cache_enabled", True)

# Always include "SearchType" and "SearchFunctions"; only include "Configuration Parameters"
# when the show_config_params_in_UI flag is True.
if configs["retrieval"]["show_config_params_in_UI"]:
    section_options = ["SearchType", "Configuration Parameters", "SearchFunctions"]
else:
    section_options = ["SearchType", "SearchFunctions"]

sidebar_tab = st.sidebar.radio(
    "Select Section", options=section_options, key="sidebar_tab"
)

# Placeholder for the dynamic content so everything stays in the sidebar area.
content_placeholder = st.sidebar.empty()

# --- SearchType tab ---
if sidebar_tab == "SearchType":
    with content_placeholder.container():
        search_options = ["Standard", "DeepSearch"]
        # Look at st.session_state.search_type to set the default index
        if "search_type" in st.session_state:
            default_index = search_options.index(
                st.session_state.search_type
            )
        else:
            default_index = 0
        selected_option = st.radio(
            "Select a search type:",
            options=search_options,
            index=default_index,
            key="search_option",
        )
        # Store the selection for downstream use
        st.session_state.search_type = selected_option
        # Update any dependent values
        st.session_state.max_return_docs = 15


# --- Configuration Parameters tab ---
elif sidebar_tab == "Configuration Parameters":
    with content_placeholder.container():
        st.header("Configuration Parameters")
        modified_config = st.session_state.modified_config
        for section, params in modified_config.items():
            with st.expander(section, expanded=False):
                if isinstance(params, dict):
                    for key, value in params.items():
                        widget_key = f"{section}_{key}"
                        if isinstance(value, int):
                            new_val = st.number_input(f"{key}", value=value, step=1, key=widget_key)
                        elif isinstance(value, float):
                            new_val = st.number_input(f"{key}", value=value, format="%.2f", key=widget_key)
                        elif isinstance(value, bool):
                            new_val = st.checkbox(f"{key}", value=value, key=widget_key)
                        else:
                            new_val = st.text_input(f"{key}", value=str(value), key=widget_key)
                        modified_config[section][key] = new_val
                else:
                    st.write(params)
        if st.button("Reset to Default", key="reset_config"):
            st.session_state.modified_config = copy.deepcopy(default_config)
            st.session_state.configs = st.session_state.modified_config
            st.success("Configuration reset to default values.")

# --- SearchFunctions tab (retrieval and early stopping controls) ---
elif sidebar_tab == "SearchFunctions":
    # Flags from config to control whether the user can toggle search methods and early stopping
    enable_search_type_selection = configs["retrieval"].get("enable_search_type_selection", False)
    enable_user_early_stopping_selection = configs["retrieval"].get("enable_user_early_stopping_selection", False)

    # When selection is disabled, force all to True
    if not enable_search_type_selection:
        st.session_state.tavily_selected = True
        st.session_state.faiss_selected = True
        st.session_state.heyligo_selected = True
        st.session_state.bm25_selected = True
        st.session_state.cache_enabled = True
        

    # If early stopping control is disabled, force it on
    if not enable_user_early_stopping_selection:
        st.session_state.early_stop_enabled = True

    # Display the controls for the SearchFunctions tab
    with content_placeholder.container():
        # Retrieval engines selection
        if enable_search_type_selection:
            st.subheader("Retrieval Engines")
            search_options = [
                ("tavily_selected", "Use Web Search", "use_tavily_web_search_cb"),
                ("faiss_selected",  "Use Semantic Search", "use_faiss_semantic_search_cb"),
                ("heyligo_selected", "Use Logbook Search", "use_heyligo_search_cb"),
                ("bm25_selected",   "Use Keyword Search", "use_bm25_lexical_search_cb"),
                ("cache_enabled",   "Use Cached Answers", "use_cached_search_cb"),
            ]
            # generate checkbox
            for var_name, label, key_name in search_options:
                st.session_state[var_name] = st.checkbox(
                    label,
                    value=st.session_state.get(var_name, False),
                    key=key_name
                )
        else:
            st.caption("All retrieval engines are enabled (fixed by configuration).")

        # Add a bit of vertical spacing
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Early stopping selection
        if enable_user_early_stopping_selection:
            st.subheader("Execution")
            st.session_state.early_stop_enabled = st.checkbox(
                "Early stopping after finding a sufficient answer",
                value=st.session_state.early_stop_enabled,
                key="enable_early_stopping_cb",
            )
        else:
            st.session_state.early_stop_enabled = True
            st.caption("Early stopping is enforced (fixed by configuration).")


############## GROQ ##############

os.environ["GROQ_API_KEY"] = configs['retrieval']['GROQ_API_KEY']
# Initialize the ChatGroq instance

# Define a wrapper class that extends ChatGroq
class GroqLLMWrapper(ChatGroq):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, human_message, system_prompt=""):
        messages = []
        if system_prompt:
            messages.append(("system", system_prompt))
        messages.append(("human", human_message))
        ai_msg = self.invoke(messages)
        return ai_msg.content
    
st.session_state.GROQ_TIMEOUT_MAX_SECONDS = configs['retrieval']['GROQ_TIMEOUT_MAX_SECONDS']
st.session_state.GROQ_TOKENS_TFM_CHECK = configs['retrieval']['DEEPSEARCH_ENABLE_GROQ_TOKENS_PER_MINUTE_CHECK']
st.session_state.GROQ_TIMEOUT_MAX_TRIES = configs['retrieval']['GROQ_TIMEOUT_MAX_TRIES']


##########################


def ensure_shared_http_client():
    if "groq_httpx" not in st.session_state:
        base = float(st.session_state.GROQ_TIMEOUT_MAX_SECONDS)

        st.session_state.groq_httpx = httpx.Client(
            timeout=httpx.Timeout(
                connect=5.0,        # quick DNS/TCP fail
                read=base * 4,      # <-- key: tolerate tail TTFB/completions
                write=base * 6,     # large prompt uploads
                pool=base * 1.5     # pool acquisition window
            ),
            limits=httpx.Limits(
                max_connections=64,
                max_keepalive_connections=32,
                keepalive_expiry=30.0,
            ),
        )
        atexit.register(st.session_state.groq_httpx.close)

ensure_shared_http_client()

init_environment()

faiss_embeddings = faiss_embeddings_init()

system = platform.system()
if not system == "Windows":
    RAG = ragatouille_init()


loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)



# Dataset choose
faiss_persist_directory = f"./faiss/{configs['data']['datasets']}/"

if 'faiss_persist_directory' not in st.session_state:
    st.session_state.faiss_persist_directory = faiss_persist_directory



# Import authentication and persistence helpers from the separate module
from libs.auth_db import (
    init_db,
    get_or_create_user,
    create_conversation,
    get_conversations,
    get_messages,
    save_message,
    update_conversation_title,
    delete_conversation,
)

def _normalize_user():
    """Return a dict-like user object with a reliable 'is_logged_in' flag,
    working for both st.user and st.experimental_user."""
    obj = getattr(st, "user", None) or getattr(st, "experimental_user", None)

    # Start with a dict of the raw data if possible
    data = {}
    if obj is None:
        pass
    elif isinstance(obj, dict):
        data = dict(obj)
    else:
        # Try best-effort conversion to dict
        try:
            data = dict(obj)
        except Exception:
            # As a last resort, copy selected attributes
            for key in ("email", "name", "username", "idp_name", "affiliation"):
                if hasattr(obj, key):
                    data[key] = getattr(obj, key)

    # Determine login state:
    # 1) Prefer the explicit flag if available (newer st.user)
    # 2) Otherwise infer from presence of identity fields
    if hasattr(obj, "is_logged_in"):
        is_logged_in = bool(getattr(obj, "is_logged_in"))
    else:
        is_logged_in = bool(data.get("email") or data.get("name") or data.get("username"))

    data["is_logged_in"] = is_logged_in
    return data

user_info = _normalize_user()

# If the user is not logged in and not already in guest mode, offer login or guest access
if not user_info.get("is_logged_in", False) and not st.session_state.get("is_guest", False):
    # Create three equal-width columns for the three buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        guest_clicked = st.button("Continue as Guest (history displayed)")        
    with col2:
        private_guest_clicked = st.button("Continue as Private Guest (no history)")
    with col3:
        # Store the clicked state so you can use it if needed
        login_clicked = st.button("Log in with LIGO credentials (not active yet)", on_click=getattr(st, "login", None))

    if private_guest_clicked:
        st.session_state["is_guest"] = True
        st.session_state["guest_private"] = True
        st.session_state.chat_history = []  # start with an empty chat
        st.session_state.current_conv_id = None
        st.rerun()
    elif guest_clicked:
        st.session_state["is_guest"] = True
        st.session_state["guest_private"] = False
        st.rerun()
    # Stop execution until one of the buttons triggers a rerun or the user logs in
    st.stop()

# The user is logged in – user_info provides claims returned by CILogon
# Example: restrict access to LIGO members based on email domain
email = (user_info.get("email") or "").lower()
# Enforce the @ligo.org domain requirement only for authenticated users
if not st.session_state.get("is_guest", False) and not email.endswith("@ligo.org"):
    st.error("LIGO Authentication is under testing. Please continue as private Guest, or switch to normal Guest via logout.")
    # Offer a logout to switch identities
    if st.button("Log out"):
        # Fallback: call st.logout if present
        logout_fn = getattr(st, "logout", None)
        if callable(logout_fn):
            logout_fn()
        st.stop()

# Optionally check affiliation or eppn from org.cilogon.userinfo scope
user_affiliation = user_info.get("affiliation", "")  # e.g., 'member@ligo.org;...'
# Greeting & details: override for guest modes
if st.session_state.get("is_guest", False):
    user_display_name = "Guest"
    user_email = None
    user_identity_provider = "Anonymous"
else:
    user_display_name = user_info.get("name") or user_info.get("email") or "user"
    user_email = user_info.get("email")
    user_identity_provider = user_info.get("idp_name")

#st.markdown(
#    f"### Welcome, {user_display_name}!  &nbsp;|&nbsp;  **Email:** {user_email}  &nbsp;|&nbsp;  **Identity provider:** {user_identity_provider}",
#    unsafe_allow_html=True
#)

st.markdown(f"### Welcome, {user_display_name}",unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Conversation selection and persistence integration
# ---------------------------------------------------------------------------
init_db()
effective_email = user_email or user_display_name
db_user_id = get_or_create_user(effective_email, user_display_name)
st.session_state["db_user_id"] = db_user_id

def _load_conversation_list():
    rows = get_conversations(st.session_state["db_user_id"])
    convs = [{"id": cid, "title": title or "", "created_at": created_at} for cid, title, created_at in rows]
    # Separate conversations with a real title from those without
    titled_convs = [c for c in convs if c["title"].strip()]
    untitled_convs = [c for c in convs if not c["title"].strip()]
    # Because get_conversations orders by created_at DESC, the recency order is preserved in each group
    return titled_convs + untitled_convs

# Auto-generate a title for an untitled conversation (<= 6 words) using recent queries
def auto_generate_conversation_title():
    conv_id = st.session_state.get("current_conv_id")
    if not conv_id:
        return
    # Find the conversation record in the current list
    conv = next((c for c in st.session_state.conversations if c["id"] == conv_id), None)
    # Do nothing if it already has a title
    if not conv or conv["title"].strip():
        return
    # Collect user messages from the chat history
    user_messages = [m["message"] for m in st.session_state.chat_history if m["role"] == "user"]
    if not user_messages:
        return
    # Use only a few recent messages to keep the prompt concise
    context = " ".join(user_messages[-5:])
    # Ask the language model to suggest a short title
    prompt = f"Using six words or fewer, create a conversation title summarizing the main topic of these queries. Only generate the title and no other text.Here are queries: \n{context}"
    candidate_title = st.session_state.llm(prompt).strip()
    # Update the title in the database if we got one back
    if candidate_title:
        update_conversation_title(conv_id, candidate_title)
        # Reload the conversation list so the new title appears next time
        st.session_state.conversations = _load_conversation_list()


# Chat history storage
st.session_state.setdefault("chat_history", [])

# Private guest: dont load recent conversation
if st.session_state.get("is_guest", False) and st.session_state.get("guest_private", False):
    
    
    # No conversation ID and no DB calls; use in-memory chat_history only
    st.session_state.current_conv_id = None
    # chat_history persists across queries in this session and will be shown in the UI
else:
    # Always reload the conversation list on each run
    st.session_state.conversations = _load_conversation_list()

    with st.sidebar:
        st.subheader("Conversations")

        # At the top of the sidebar, before rendering the selectbox
        if st.session_state.get("select_new_conv"):
            # New conversation or deletion: select the new conversation or default to "new"
            current_conv = st.session_state.get("current_conv_id")
            st.session_state.selected_conv_id = (
                str(current_conv) if current_conv is not None else "new"
            )
            st.session_state.pop("select_new_conv")
        elif st.session_state.get("refresh_selected_conv"):
            # Rename: reselect the current conversation by ID
            current_conv = st.session_state.get("current_conv_id")
            if current_conv is not None:
                st.session_state.selected_conv_id = str(current_conv)
            st.session_state.pop("refresh_selected_conv")


        # Build select options
        options = [("new", "➕ New conversation")]
        for conv in st.session_state.conversations:
            label = conv["title"].strip() or "Untitled conversation"
            options.append((str(conv["id"]), label))
        selected_key = st.selectbox(
            "Select a conversation",
            options=[opt[0] for opt in options],
            format_func=lambda k: next(label for key, label in options if key == k),
            key="selected_conv_id",
        )

        if selected_key == "new":
            new_conv_id = create_conversation(st.session_state["db_user_id"], title=None)
            st.session_state.current_conv_id = new_conv_id
            st.session_state.chat_history = []
            st.session_state.select_new_conv = True
            st.rerun()
        else:
            conv_id = int(selected_key)
            st.session_state.current_conv_id = conv_id
            st.session_state.chat_history = get_messages(conv_id)

            # Display the current title in an input box
            current_title = next(
                (c["title"] for c in st.session_state.conversations if c["id"] == conv_id),
                "",
            )
            new_title = st.text_input(
                "Conversation title",
                value=current_title,
                key=f"title_input_{conv_id}",
            )

            # User edits the title
            if new_title.strip() and new_title.strip() != current_title:
                update_conversation_title(conv_id, new_title.strip())
                st.session_state.conversations = _load_conversation_list()
                st.session_state.chat_history = get_messages(conv_id)
                # Use a separate flag for renames only
                st.session_state.refresh_selected_conv = True
                st.rerun()

            # Delete conversation
            if st.button("Delete conversation", key=f"delete_conv_{conv_id}"):
                delete_conversation(conv_id)
                st.session_state.current_conv_id = None
                st.session_state.chat_history = []
                # Reload conversation list to remove deleted conversation
                st.session_state.conversations = _load_conversation_list()
                st.session_state.select_new_conv = True
                st.rerun()

            # Soft Kill button
            stop_col, info_col = st.columns([10, 6])
            stop_col.button("⛔ Stop Search", key="stop_search_btn", on_click=_request_stop_search)



######## AUTHENTICATION + CONVERSATION LOADING ENDS ###################



def is_empty_or_whitespace(s):
    return not s.strip()

class CustomBooleanOutputParser(BaseOutputParser[bool]):
    """Parse the output of an LLM call to a boolean. Default to True if response not formatted correctly."""
    true_val: str = "YES"
    """The string value that should be parsed as True."""
    false_val: str = "NO"
    """The string value that should be parsed as False."""

    def parse(self, text: str) -> bool:
        """Parse the output of an LLM call to a boolean by checking if YES/NO is contained in the output.
        Args:
            text: output of a language model.
        Returns:
            boolean
        """
        cleaned_text = text.strip().upper()
        # print(cleaned_text)
        return self.false_val not in cleaned_text

    @property
    def _type(self) -> str:
        """Snake-case string identifier for an output parser type."""
        return "custom_boolean_output_parser"


# Modified LLMChainFilter Prompt
prompt_template = "Given the following question and context, \
                   return YES if the context has some overlapping terms to the question and NO if it isn't. \
                    \n\n> Question: {question}\n> Context:\n>>>\n{context}\n>>>\n> Relevant (YES / NO)"

custom_llm_chain_filter_prompt_template = PromptTemplate(
    template=prompt_template,
    input_variables=["question", "context"],
    output_parser=CustomBooleanOutputParser(),
)


def animate_text_streaming(response_result, response_placeholder, height):
    full_response = ""
    for chunk in response_result.split():
        full_response += chunk + " "
        time.sleep(0.05)
        # Add a blinking cursor to simulate typing
        response_placeholder.text_area(full_response + "▌", height=height)
    response_placeholder.text_area(full_response, height=height)
    response_placeholder.text_area(response_result, height=height)


def generate_QApairs(llm_QApair_generator, CURRENT_CONTEXT):
    my_template = configs["prompts"]["augment_user_query_template"]["template"]
    my_prompt = PromptTemplate(input_variables=["CURRENT_CONTEXT"], template=my_template)
    augment_user_query_template = my_prompt.format(CURRENT_CONTEXT=CURRENT_CONTEXT)
    print(augment_user_query_template)
    QApairs = llm_QApair_generator(augment_user_query_template)
    return QApairs




# recursive_summarization 

def recursive_summarization(ans_list_initial, user_query, embeddings, llm):
    # Convert text chunks into vector embeddings
    def embed_text_chunks(text_chunks):
        return np.array([embeddings.embed_query(chunk) for chunk in tqdm(text_chunks, desc="Embedding text chunks")])
    # Cluster documents into N clusters using KMeans

    def cluster_documents(embedding_matrix, num_clusters):
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embedding_matrix)
        clusters = {i: [] for i in range(num_clusters)}
        for idx, label in enumerate(cluster_labels):
            clusters[label].append(ans_list_initial[idx])
        return clusters
    # Summarize a group of text chunks with focus on user_query

    def summarize_chunks(chunks, user_query):
        combined_text = " ".join(chunks)
        prompt = f"Summarize the following <TEXT> with a focus on the <QUERY>. <QUERY>: {user_query}. <TEXT>: {combined_text}"
        # Using LLM to summarize with focus on user query
        summary = llm(prompt)
        return summary.strip()
    # Recursive summarization

    def recursive_cluster_summarization(text_chunks, user_query, iteration=1, all_summaries=None):
        if all_summaries is None:
            all_summaries = []
        print(
            f"\nTreeSearch Iteration {iteration}: Number of text chunks = {len(text_chunks)}")
        # Stop when only one chunk remains
        if len(text_chunks) <= 1:
            all_summaries.append(text_chunks[0])
            # Base case: single chunk remains
            return text_chunks[0], all_summaries
        # Convert chunks to embeddings
        embeddings_matrix = embed_text_chunks(text_chunks)
        # Determine optimal number of clusters (e.g., half of the number of chunks)
        num_clusters = max(1, len(text_chunks) // 2)
        # Cluster the documents
        clusters = cluster_documents(embeddings_matrix, num_clusters)
        # Print remaining clusters for debugging
        print(f"Remaining clusters at iteration {iteration}:")
        for i, cluster in clusters.items():
            print(f"Cluster {i}: {cluster}")
        # Summarize each cluster with focus on user query
        summarized_clusters = []
        for cluster in clusters.values():
            summary = summarize_chunks(cluster, user_query)
            summarized_clusters.append(summary)
            all_summaries.append(summary)
        # Check if summarized clusters are the same as input (to avoid infinite loops)
        if summarized_clusters == text_chunks:
            print("Converged: No change in clusters. Stopping recursion.")
            # Stop recursion if no changes
            return " ".join(summarized_clusters), all_summaries
        # Recursively summarize the new summarized chunks
        return recursive_cluster_summarization(summarized_clusters, user_query, iteration + 1, all_summaries)
    # Start the recursive summarization process
    final_summary, all_summaries = recursive_cluster_summarization(
        ans_list_initial, user_query)
    return final_summary, all_summaries


parser = argparse.ArgumentParser(description='Filter out URL argument.')

st.session_state.useGroq = configs['retrieval']['enable_groq_inference_for_public_docs']

if not st.session_state.useGroq:
    # Load LLMs
    for idx, key, model_key in [
        (1, 'llm', 'primary_llm_model'),
        (2, 'llm_2', 'secondary_llm_model'),
        (3, 'llm_3', 'tertiary_llm_model'),
        (4, 'llm_4', 'quaternary_llm_model'),
        (5, 'llm_5', 'fifth_llm_model'),
        (6, 'llm_query_improve', 'primary_llm_model'),
    ]:
        if key not in st.session_state:
            st.session_state[key] = Ollama(
                base_url=configs['server']['ollama']['base_url'],
                model=configs['models'][model_key],
                verbose=configs['retrieval']['enable_verbose'],
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
            )
            print(f"\n Loaded AUX LLM_{idx} model: {getattr(st.session_state[key], 'model', None)}")
    # LLM USER_QUERY_IMPROVE_MODEL
    print(f"\n Loaded QUERY AUGMENTING LLM model: {st.session_state.llm_query_improve.model}")
        
else:
    print("Using Groq Cloud Inference API for Public Docs")
    groq_models_to_load = [
        ("llm", "primary_groq_llm_model", "PRIMARY"),
        ("llm_2", "secondary_groq_llm_model", "SECONDARY"),
        ("llm_3", "tertiary_groq_llm_model", "TERTIARY"),
        ("llm_4", "quaternary_groq_llm_model", "QUATERNARY"),
        ("llm_5", "fifth_groq_llm_model", "FIFTH"),
        ("llm_query_improve", "query_improvement_groq_model", "QUERY AUGMENTING")
    ]
    ollama_models_to_load = [
        ("ollama_llm", "primary_llm_model", "PRIMARY"),
        ("ollama_llm_2", "secondary_llm_model", "SECONDARY"),
        ("ollama_llm_3", "tertiary_llm_model", "TERTIARY")
    ]

    print("Using Groq Cloud Inference API for Public Docs")
    # Load GROQ models
    for state_key, config_key, label in groq_models_to_load:
        if state_key not in st.session_state:
            st.session_state[state_key] = GroqLLMWrapper(
                model=configs['models'][config_key],
                temperature=0,
                max_tokens=None,
                max_retries=st.session_state.GROQ_TIMEOUT_MAX_TRIES,
                http_client=st.session_state.groq_httpx,
            )
        print(f"\n Loaded API-Based {label}_LLM_MODEL: GROQ-{configs['models'][config_key]}")

    # Load Ollama models
    for state_key, config_key, label in ollama_models_to_load:
        if state_key not in st.session_state:
            st.session_state[state_key] = Ollama(
                base_url=configs['server']['ollama']['base_url'],
                model=configs['models'][config_key],
                verbose=configs['retrieval']['enable_verbose'],
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
            )
        print(f"\n Loaded Local {label}_LLM_MODEL: {st.session_state[state_key].model}")


# Now import RAG_STAR 
from libs.ragSTAR_v3_mcts_fix_citations_v2_reasoning_trace import RAG_STAR


if 'template' not in st.session_state:
    st.session_state.template = configs['prompts']['general_template']['template']

if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )



#####################
# -----------------------------------------------------------------------------
# Conversation memory + conversation-switch sync
# -----------------------------------------------------------------------------

def _make_memory():
    # return_messages=False gives you a clean string for prompts
    # output_key="text" matches LLMChain default; safe for save_context usage
    return ConversationSummaryBufferMemory(
        llm=st.session_state.llm,
        memory_key="history",
        max_token_limit=int(configs["retrieval"].get("memory_max_token_limit", 1200)),
        return_messages=False,
        input_key="question",
        output_key="text",
    )

PP_HINTS = [
    "format this","reformat","make a table","tabulate","table please",
    "only the equations","extract equations","show equations",
    "just the numbers","only numbers","only citations","only references",
    "summarize this","bulletize","make bullets","json","csv",
    "clean up","prettify","markdown only","latex only","markdown table","latex table"
]

def _looks_like_postprocess_request(u: str) -> bool:
    u = (u or "").strip().lower()
    if not u:
        return False

    # reuse your hints (already defined in your file) + add some “from above” style triggers
    extra = [
        "from above", "above", "previous answer", "earlier answer", "last answer",
        "what are we talking about", "what did we talk about", "recap", "summarize", "tl;dr",
        "links from above", "relevant links", "show the links"
    ]
    return any(h in u for h in PP_HINTS) or any(h in u for h in extra)

def _is_significant_assistant_text(a: str) -> bool:
    a = (a or "").strip()
    if len(a) < 200:          # tune threshold if needed
        return False
    # avoid common short boilerplate patterns
    low = a.lower()
    if low in {"ok", "okay", "sure"}:
        return False
    if low.startswith("let me know what you would like to know"):
        return False
    return True

def _set_pp_prev_significant_answer_from_history() -> None:
    """
    Pick the last assistant message that looks like a real “answer” turn,
    not a postprocess/recap turn.
    """
    sig = None
    pending_user = None

    for msg in st.session_state.chat_history:
        role = msg.get("role")
        text = (msg.get("message") or "").strip()
        if not text:
            continue

        if role == "user":
            pending_user = text

        elif role == "assistant":
            if _is_significant_assistant_text(text) and not _looks_like_postprocess_request(pending_user or ""):
                sig = text
            pending_user = None

    # fallback
    st.session_state["_pp_prev_significant_answer"] = sig or st.session_state.get("_pp_prev_answer")


def _pp_debug_print_memory_state():
    summary = ""
    mem = st.session_state.get("memory")
    if mem:
        try:
            summary = (mem.load_memory_variables({}).get("history") or "").strip()
        except Exception:
            # fallback for some langchain versions
            try:
                summary = (mem.load_memory_variables({"question": ""}).get("history") or "").strip()
            except Exception:
                summary = ""

if "memory" not in st.session_state:
    st.session_state.memory = _make_memory()

def _set_pp_prev_answer_from_history():
    last = next(
        (m["message"] for m in reversed(st.session_state.chat_history)
         if m.get("role") == "assistant"),
        None
    )
    if last:
        st.session_state["_pp_prev_answer"] = last

def _reset_query_rewrite_state():
    st.session_state.stage = "initial"
    st.session_state.updated_query_for_confirm = None
    st.session_state.concatenated_definitions_list = []
    st.session_state.current_query = ""

def _rebuild_memory_from_chat_history():
    """Re-hydrate ConversationSummaryBufferMemory from the currently loaded chat_history."""
    mem = st.session_state.get("memory")
    if mem is None:
        return

    # Clear existing memory content
    try:
        mem.clear()
    except Exception:
        st.session_state.memory = _make_memory()
        mem = st.session_state.memory

    pending_user = None
    for msg in st.session_state.chat_history:
        role = msg.get("role")
        text = (msg.get("message") or "").strip()
        if not text:
            continue

        if role == "user":
            pending_user = text

        elif role == "assistant":
            q = pending_user or ""
            try:
                mem.save_context({"question": q}, {"text": text})
            except Exception:
                # fallback if keys differ in your langchain version
                mem.save_context({"input": q}, {"output": text})
            pending_user = None

def ensure_runtime_synced_to_loaded_conversation():
    """
    Call this once per rerun. If the user selected a different conversation in the sidebar:
    - reset query rewrite state machine
    - restore _pp_prev_answer
    - rebuild memory from loaded chat_history
    """
    if "_active_conv_id" not in st.session_state:
        st.session_state._active_conv_id = "__INIT__"

    conv_id = st.session_state.get("current_conv_id")
    if st.session_state._active_conv_id != conv_id:
        st.session_state._active_conv_id = conv_id
        _reset_query_rewrite_state()
        _set_pp_prev_answer_from_history()
        _set_pp_prev_significant_answer_from_history()  
        _rebuild_memory_from_chat_history()

# Call it once per run (after chat_history is loaded from DB and memory exists)
ensure_runtime_synced_to_loaded_conversation()

print_wide_line_message("After ensure_runtime_synced_to_loaded_conversation()")
_pp_debug_print_memory_state()


def maybe_save_turn_to_memory(question: str, answer: str) -> None:
    mem = st.session_state.get("memory")
    if not mem:
        return
    q = (question or "").strip()
    a = (answer or "").strip()
    if not q or not a:
        return

    # Optional: de-dupe if the last stored AI message equals this answer
    try:
        msgs = getattr(getattr(mem, "chat_memory", None), "messages", None)
        if msgs:
            last = msgs[-1]
            last_content = getattr(last, "content", None)
            if isinstance(last_content, str) and last_content.strip() == a:
                return
    except Exception:
        pass

    try:
        mem.save_context({"question": q}, {"text": a})
    except Exception:
        mem.save_context({"input": q}, {"output": a})

#####################

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


def create_llm(base_url, model, verbose):
    return Ollama(
        base_url=base_url,
        model=model,
        verbose=verbose,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )


st.session_state.LIGO_ACRONYMS = load_acronyms(configs['retrieval']['LIGO_ACRONYM_FILE'])

st.write(st.session_state.llm(
    f'''
    Rephrase the following in English using a single line. Donot add anything else.
    Hello ! Great to talk to you :) What would you like to know about LIGO and gravitational waves?. 
    '''))

# Default retrieval doc count
st.session_state.initial_retrieval_count = configs['retrieval']['initial_retrieval_count']
st.session_state.max_return_docs         = configs['retrieval']['max_return_docs']


for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])
        # If this assistant message has an attached reasoning trace, show it in an expander
        if isinstance(message, dict) and message.get("trace"):
            try:
                render_reasoning_trace_expander(message["trace"])
            except Exception:
                pass

# QUERY IMPROVEMENT


def is_empty_or_whitespace(s):
    """Check if a string is empty or contains only whitespace."""
    return len(s.strip()) == 0


# State initialization
if 'stage' not in st.session_state:
    st.session_state.stage = 'initial'
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_query' not in st.session_state:
    st.session_state.current_query = ''


def get_chat_history_string(num_last_lines=6):
    chat_history_string_full = "\n".join([f"{msg['role']}: {msg['message']}" for msg in st.session_state.chat_history])
    # Split the input text into lines and filter
    chat_history_string_last_few_lines = [line for line in chat_history_string_full.splitlines(
    ) if line.startswith('user:') or line.startswith('assistant:')]
    chat_history_string_last_few_lines = chat_history_string_last_few_lines[-num_last_lines:-1]
    chat_history_string_last_few_lines = "\n".join(chat_history_string_last_few_lines)
    return chat_history_string_last_few_lines

def get_working_history(num_last_lines_fallback: int = 10) -> str:
    mem = st.session_state.get("memory")
    if mem:
        try:
            hist = mem.load_memory_variables({}).get("history", "")
            if isinstance(hist, str) and hist.strip():
                return hist.strip()
        except Exception as e:
            print(f"[memory] load_memory_variables failed: {e}")

    return get_chat_history_string(num_last_lines=num_last_lines_fallback)




def append_query_with_acronym(user_input):
    # Initialize the prompt template for acronym detection
    llm_acronym_check = PromptTemplate(
        input_variables=["user_query"],
        template=configs['prompts']['DETECT_ACRONYM_template']['template']
    )
    
    # Get the LLM response containing detected acronyms
    llm_response = st.session_state.llm(llm_acronym_check.format(user_query=user_input))    
    # Split the response into individual acronyms and strip whitespace and punctuation
    detected_acronyms = [
        acro.strip(string.whitespace + string.punctuation) 
        for acro in llm_response.split(',') 
        if acro.strip()
    ]    
    print(f"\nDetected Acronyms: {detected_acronyms} for the USER_INPUT: {user_input}")    
    # Initialize a list to collect definitions
    definitions = []    
    if detected_acronyms:
        # Create a case-insensitive mapping of LIGO_ACRONYMS
        ligo_acronyms_lower = {k.lower(): v for k, v in st.session_state.LIGO_ACRONYMS.items()}        
        for acro in detected_acronyms:
            acro_lower = acro.lower()            
            # Skip the acronym if it's 'ligo' (case-insensitive)
            if acro_lower != "ligo":
                # Retrieve the definition using the lowercase acronym
                acro_definition = ligo_acronyms_lower.get(acro_lower)
                
                if acro_definition:
                    # Append the definition to the user input with a period separator
                    user_input = f"{user_input}. {acro_definition}"
                    definitions.append(acro_definition)  # Collect the definition
                    print(f"Appended Query: {user_input}")
                else:
                    print(f"Definition for acronym '{acro}' not found.")    
    # Concatenate all collected definitions into a single string separated by spaces
    concatenated_definitions = ' '.join(definitions)
    # Optionally, you can also log the concatenated definitions
    print(f"Concatenated Definitions: {concatenated_definitions}")
    # Return both the modified user_input and the concatenated definitions
    return user_input, concatenated_definitions


# ─────────────────────────────────────────────────────────────────────────────
#  process_user_query_and_switch_states  —  with smart auto-confirm
# ─────────────────────────────────────────────────────────────────────────────


# ─── tiny helpers ────────────────────────────────────────────────────────────
_WRAP_RE = re.compile(
    r"""
    ^\s*
    (?:did\s+you\s+mean\s+to\s+ask\s*:?\s*)     # “Did you mean to ask …”
    (["“”']?)                                   # optional opening quote
    (.+?)                                       # the real question
    \1                                         # matching closing quote (if any)
    [\s\.\?]*$                                  # trailing dot / space / “?”
    """,
    re.I | re.X,
)

def _unwrap(text: str) -> str:
    """Remove ‘Did you mean to ask: …’ wrappers if present."""
    m = _WRAP_RE.match(text.strip())
    return m.group(2) if m else text

def _clean(text: str) -> str:
    """Lower-case, strip punctuation, squeeze spaces → quick normaliser."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()

def queries_equivalent(q1: str, q2: str, thresh: float = 0.88) -> bool:
    """
    True if q1 and q2 are essentially the same sentence.
    * Unwraps ‘Did you mean…’ boiler-plate first.
    * Uses difflib.SequenceMatcher on lightly normalised strings.
    """
    q1_core = _clean(_unwrap(q1))
    q2_core = _clean(_unwrap(q2))
    return difflib.SequenceMatcher(None, q1_core, q2_core).ratio() >= thresh
# ─────────────────────────────────────────────────────────────────────────────


def process_user_query_and_switch_states() -> Tuple[Optional[str], str]:
    """
    Conversational state-machine that:
      • improves vague queries with an LLM,
      • skips confirmation when the 'updated' query is basically the same,
      • handles acronym expansion, confused loops, etc.
    Returns (final_query or None, concatenated_acronym_definitions)
    """
    print("Executing process_user_query_and_switch_states")

    # Prompt templates (unchanged)
    llm_generate_meaningful_query_prompt = PromptTemplate(
        input_variables=["history", "user_input"],
        template=configs["prompts"]["user_query_template"]["template"],
    )
    llm_check_user_reaction_prompt = PromptTemplate(
        input_variables=["user_reaction"],
        template=configs["prompts"]["user_reaction_template"]["template"],
    )
    llm_final_query_check = PromptTemplate(
        input_variables=["user_query"],
        template=configs["prompts"]["query_content_template"]["template"],
    )

    # scratch-pad for acronym expansions
    if "concatenated_definitions_list" not in st.session_state:
        st.session_state.concatenated_definitions_list = []
    # scratch-pad to store a candidate query between states
    if "updated_query_for_confirm" not in st.session_state:
        st.session_state.updated_query_for_confirm = None

    user_input = st.session_state.user_input

    # Empty/whitespace guard (unchanged)
    if is_empty_or_whitespace(user_input):
        msg = "Let me know what you would like to know."
        streamlit_add_msg(st=st, role="assistant", message=msg, persist=True)
        return None, ""

    # ── STAGE: initial ─────────────────────────────────────────────────────
    if st.session_state.stage == "initial":
        print("State: initial")
        user_input = st.session_state.chat_history[-1]["message"]

        # Acronym expansion (unchanged)
        user_input, defs0 = append_query_with_acronym(user_input)
        if defs0:
            st.session_state.concatenated_definitions_list.append(defs0)

        # LLM proposes improved query (unchanged)
        updated_query = st.session_state.llm_query_improve(
            llm_generate_meaningful_query_prompt.format(
                history=get_working_history(),
                user_input=user_input,
            )
        ).strip()

        # Auto-confirm if nothing meaningful changed (unchanged)
        if queries_equivalent(user_input, updated_query):
            concat_defs = " ".join(st.session_state.concatenated_definitions_list).strip()
            st.session_state.concatenated_definitions_list = []
            st.session_state.stage = "initial"
            return updated_query, concat_defs

        # Otherwise ask the user for confirmation
        # Save the candidate query for later retrieval
        st.session_state.updated_query_for_confirm = updated_query

        # Display the assistant's response and persist it (the prefix helps user understand)
        streamlit_add_msg(
            st=st,
            role="assistant",
            message=f"Updated Query: {updated_query}.",
            persist=True
        )

        # DO NOT append updated_query separately – persist=True above already wrote to chat_history

        st.session_state.stage = "user_reaction"
        print("Switching state to: user_reaction")
        return None, " ".join(st.session_state.concatenated_definitions_list).strip()

    # ── STAGE: user_reaction ────────────────────────────────────────────────
    elif st.session_state.stage == "user_reaction":
        print("State: user_reaction")
        user_reply = st.session_state.chat_history[-1]["message"]

        # Acronym expansion (unchanged)
        user_reply, defs1 = append_query_with_acronym(user_reply)
        if defs1:
            st.session_state.concatenated_definitions_list.append(defs1)

        # Retrieve the candidate query saved in the previous step; fallback to history if absent
        candidate_query = st.session_state.updated_query_for_confirm
        if not candidate_query:
            history = st.session_state.chat_history
            if len(history) >= 2:
                candidate_query = history[-2]["message"]
            else:
                candidate_query = ""

        # Auto-confirm if the user re-typed essentially the same query (unchanged)
        if queries_equivalent(user_reply, candidate_query):
            concat_defs = " ".join(st.session_state.concatenated_definitions_list).strip()
            st.session_state.concatenated_definitions_list = []
            st.session_state.stage = "initial"
            st.session_state.updated_query_for_confirm = None
            return candidate_query, concat_defs

        # Ask LLM if the reaction was affirmative (unchanged)
        reaction_flag = st.session_state.llm_query_improve(
            llm_check_user_reaction_prompt.format(user_reaction=user_reply)
        ).strip().lower()

        if "true" in reaction_flag:
            msg = "I understand what you're looking for. Let me search through the resources and find the answer."
            streamlit_add_msg(
                st=st,
                role="assistant",
                message=msg,
                persist=True
            )
            concat_defs = " ".join(st.session_state.concatenated_definitions_list).strip()
            st.session_state.concatenated_definitions_list = []
            st.session_state.stage = "initial"
            # Clear out the stored candidate for next time
            st.session_state.updated_query_for_confirm = None
            return candidate_query, concat_defs

        if "false" in reaction_flag:
            msg = "I'm sorry, I am still trying to understand your question. Can you explain a bit more?"
            streamlit_add_msg(
                st=st,
                role="assistant",
                message=msg,
                persist=True
            )
            st.session_state.stage = "initial"
            concat_defs = " ".join(st.session_state.concatenated_definitions_list).strip()
            st.session_state.concatenated_definitions_list = []
            st.session_state.updated_query_for_confirm = None
            return None, concat_defs

        # ------------------------------------------------------------------
        # Fallback: if user didn't clearly say yes/no, they might be EDITING
        # the query (e.g., "also include citations", "focus on X", etc.).
        # ------------------------------------------------------------------

        # 1) If user_reply itself is a usable standalone query, accept it.
        final_check = st.session_state.llm_query_improve(
            llm_final_query_check.format(user_query=user_reply)
        ).strip().lower()

        if ("true" in final_check) or ("yes" in final_check):
            concat_defs = " ".join(st.session_state.concatenated_definitions_list).strip()
            st.session_state.concatenated_definitions_list = []
            st.session_state.stage = "initial"
            st.session_state.updated_query_for_confirm = None
            return user_reply, concat_defs

        # 2) If it's likely a "modifier/edit" to the candidate query, merge it
        #    instead of entering the confused loop.
        edit_prefixes = ("also", "and", "add", "include", "exclude", "remove", "focus", "only", "but", "plus")
        if candidate_query and user_reply.strip().lower().startswith(edit_prefixes) and len(user_reply.split()) >= 3:
            edited_query = f"{candidate_query}. {user_reply}".strip()

            concat_defs = " ".join(st.session_state.concatenated_definitions_list).strip()
            st.session_state.concatenated_definitions_list = []
            st.session_state.stage = "initial"
            st.session_state.updated_query_for_confirm = None
            return edited_query, concat_defs


        # ── confused loop (unchanged except storing the new candidate) ─────
        print("User seems confused. Asking for clarification.")
        st.session_state.stage = "confused"
        user_reply, defs2 = append_query_with_acronym(user_reply)
        if defs2:
            st.session_state.concatenated_definitions_list.append(defs2)

        updated_query = st.session_state.llm_query_improve(
            llm_generate_meaningful_query_prompt.format(
                history=get_working_history(),
                user_input=user_reply,
            )
        ).strip()

        msg = f"Updated Query: {updated_query}"
        streamlit_add_msg(
            st=st,
            role="assistant",
            message=msg,
            persist=True
        )
        # Save this new suggestion for the next user reaction
        st.session_state.updated_query_for_confirm = updated_query
        st.session_state.stage = "user_reaction"
        return None, " ".join(st.session_state.concatenated_definitions_list).strip()

    # Fallback (unchanged)
    print("Invalid state encountered. Resetting to 'initial'.")
    st.session_state.stage = "initial"
    concat_defs = " ".join(st.session_state.concatenated_definitions_list).strip()
    st.session_state.concatenated_definitions_list = []
    st.session_state.updated_query_for_confirm = None
    return None, concat_defs



def contains_word(string, word):
    # Convert both the string and word to lowercase for case-insensitive comparison
    return word.lower() in string.lower()


def filter_documents_by_relevance(documents, threshold=15):
    """
    Filters a list of Document objects based on the 'relevance_score' metadata.
    Args:
    - documents (list): A list of Document objects to be filtered.
    - threshold (float): The relevance score threshold. Defaults to 15.
    Returns:
    - list: A filtered list of Document objects that have 'relevance_score' 
            metadata and that score is greater than or equal to the threshold.
    """
    filtered_docs = []
    # Iterate over each document in the list
    for doc in documents:
        # Check if 'relevance_score' exists in the document's metadata
        if 'relevance_score' in doc.metadata:
            # Check if the relevance score is greater than or equal to the threshold
            if doc.metadata['relevance_score'] >= threshold:
                filtered_docs.append(doc)
    # If filtered_docs is not empty, return the filtered list
    if filtered_docs:
        return filtered_docs
    # If no documents matched the filter, return the original list
    return documents





def diagnostic_print_RAG_chunks(user_input, max_return_docs):
    print_wide_line_message("BM25+ DOC SIMILARITY SEARCH RESULTS")
    try:
        XYZ = st.session_state.bm25_retriever.get_relevant_documents(
            user_input)
        print_langchain_documents(XYZ)
    except:
        pass
    print_wide_line_message("FAISS DOC SIMILARITY SEARCH RESULTS")
    try:
        XYZ = st.session_state.FAISS_retriever.get_relevant_documents(
            user_input, k=st.session_state.max_return_docs)
        print_langchain_documents(XYZ)
    except:
        pass
    print_wide_line_message("DuckDuckGo DOC SIMILARITY SEARCH RESULTS")
    try:
        XYZ = enhanced_retriever_duckduckgo(
            user_input, max_return_docs=st.session_state.max_return_docs)
        print_langchain_documents(XYZ)
    except:
        pass


def get_search_specific_relevance_score(
    user_query: str,
    baseline_answer: str,
    *,
    num_repeats: int = 5
) -> float:
    """
    Ask the LLM `num_repeats` times to rate how well *baseline_answer*
    answers *user_query*.  
    The prompt must output a real number 0‒1, preferably wrapped in
    <SCORE>…</SCORE>.  Scores that cannot be parsed are ignored (logged).

    Returns the **mean** of all successfully parsed scores.  If none could
    be parsed the function returns 0.0.
    """

    # ---------- build prompt once ------------------------------------------
    template = configs['prompts']['ASSESS_ANSWER_RELEVANCE_template']['template']
    prompt   = PromptTemplate(
        input_variables=["user_query", "generated_response"],
        template=template,
    )

    # ---------- helpers ----------------------------------------------------
    def _try_float(txt: str) -> Optional[float]:
        """Return float(txt) if 0 ≤ txt ≤ 1, else None."""
        try:
            val = float(txt)
            return val if 0.0 <= val <= 1.0 else None
        except Exception:
            return None

    def _extract_score(reply: str) -> Optional[float]:
        """
        Try multiple patterns until a legal 0–1 number is found.
        Supports:
            <SCORE>0.83</SCORE>
            score: 0.75
            SCORE =0.9
            bare 0.6
        """
        # 1) <SCORE> … </SCORE>
        m = re.search(r'<score[^>]*>\s*([0-9]*\.?[0-9]+)\s*</score>', reply, re.I)
        if m:
            return _try_float(m.group(1))

        # 2) score: 0.87   or   SCORE = 0.8
        m = re.search(r'score\s*[:=]\s*([0-9]*\.?[0-9]+)', reply, re.I)
        if m:
            return _try_float(m.group(1))

        # 3) any bare number between 0 and 1
        for num in re.findall(r'[0-9]*\.?[0-9]+', reply):
            val = _try_float(num)
            if val is not None:
                return val
        return None
    # ----------------------------------------------------------------------

    total, ok = 0.0, 0
    for i in range(num_repeats):
        formatted = prompt.format(
            user_query=user_query,
            generated_response=baseline_answer
        )
        raw_reply = st.session_state.llm(formatted)
        score     = _extract_score(raw_reply)

        if score is None:
            print(f"[relevance {i+1}/{num_repeats}] ❌ could not parse score → 0.0")
            continue

        print(f"[relevance {i+1}/{num_repeats}] ✅ score={score}")
        ok    += 1
        total += score

    if ok == 0:
        print("‼️  All relevance checks failed – returning 0.0")
        return 0.0

    avg = total / ok
    print(f"→ average relevance score over {ok} successful runs = {avg:.3f}")
    return avg



def ensembele_superposition_answer(
    user_query: str,
    enable_search_type_selection: bool = True,
    enable_user_early_stopping_selection: bool = True,
) -> Tuple[str, str, str, str, List[Any], Any]:
    """Retrieve relevant documents from multiple sources, optionally allowing
    Streamlit users to choose which sources to query and whether to employ
    early stopping.

    Args:
        user_query: The question posed by the user.
        st: A reference to the ``streamlit`` module (typically imported as ``st``)
            providing access to session state and widget helpers.
        enable_search_type_selection: When ``True``, expose checkboxes in the
            UI for selecting individual retrieval strategies.  When ``False``,
            all four retrieval strategies (Tavily web, FAISS semantic, HeyLIGO,
            and BM25 lexical) are executed.
        enable_user_early_stopping_selection: When ``True``, expose a
            checkbox allowing the user to disable early stopping.  When
            ``False``, early stopping remains enabled and mirrors the
            behaviour of the original implementation.

    Returns:
        A 6‑tuple consisting of ``FINAL_RESPONSE``, ``QA_ANSWER``,
        ``WIKI_SUMMARY``, ``MORE_DETAILS``, the list of filtered retrieved
        documents that were deemed relevant by the verification stage, and the
        aggregator object.  The semantics of these values are unchanged from
        the original ``ensembele_superposition_answer``.
    """
    # Notify the user that the search is starting.  This replicates the
    # behaviour of the original function and preserves the wide line message
    # semantics.
    print_wide_line_message("Executing SuperPosition Search")

    # Initialise a document counter within the Streamlit session state.  This
    # counter is used elsewhere in the system to label retrieved documents.
    st.session_state.doc_counter = 1

    # Trace: high-level retrieval plan for this call
    try:
        _sources = []
        if st.session_state.get("tavily_selected"):
            _sources.append("web")
        if st.session_state.get("faiss_selected"):
            _sources.append("semantic")
        if st.session_state.get("heyligo_selected"):
            _sources.append("heyligo")
        if st.session_state.get("bm25_selected"):
            _sources.append("bm25")
        trace_add(
            "Superposition retrieval",
            detail=(
                f"Query: {user_query[:300]}\n"
                f"Sources: {', '.join(_sources) if _sources else 'none'}\n"
                f"Early stop: {bool(st.session_state.get('early_stop_enabled', True))}"
            )
        )
    except Exception:
        pass

    # A context manager to display a spinner while documents are being
    # retrieved.  This replicates the user feedback of the original function.
    with traced_spinner("Reading relevant documents", detail="Starting retrieval from selected sources."):
        # Work around LangChain embedding bug by ensuring a 'headers'
        # attribute exists on the embeddings object.  This patch is identical
        # to the original implementation and should remain in place until the
        # underlying issue is resolved in LangChain.
        try:
            if not hasattr(st.session_state.vectorstore.embeddings, "headers"):
                setattr(
                    st.session_state.vectorstore.embeddings,
                    "headers",
                    {"X-Fake-Header": "RandomValue"},
                )
        except Exception:
            pass




        # Prepare containers for documents retrieved from each source.  These
        # variables mirror the naming in the original implementation and allow
        # for clean aggregation later.  For each method we maintain both the
        # raw documents returned from the retriever and those documents that
        # passed the LLM based verification step.
        docs_Tavily: List[Any] = []
        docs_FAISS: List[Any] = []
        docs_HeyLIGO: List[Any] = []
        docs_BM25: List[Any] = []
        filtered_retrieved_docs_Tavily: List[Any] = []
        filtered_retrieved_docs_FAISS: List[Any] = []
        filtered_retrieved_docs_HeyLIGO: List[Any] = []
        filtered_retrieved_docs_BM25: List[Any] = []
        filtered_retrieved_docs_LLM_answers_Tavily: List[Any] = []
        filtered_retrieved_docs_LLM_answers_FAISS: List[Any] = []
        filtered_retrieved_docs_LLM_answers_HeyLIGO: List[Any] = []
        filtered_retrieved_docs_LLM_answers_BM25: List[Any] = []

        # Determine whether the query is subject to the strict logbook mode
        # defined by the application prompts.  If strict logbook is True
        # certain retrieval methods should be bypassed entirely.  The logic
        # remains unchanged from the original implementation.
        my_template = configs["prompts"]["STRICT_LOGBOOK_FLAG_template"]["template"]
        my_prompt = PromptTemplate(input_variables=["user_query"], template=my_template)
        my_formatted_prompt = my_prompt.format(user_query=user_query)
        my_out = st.session_state.llm(my_formatted_prompt)
        strict_logbook_FLAG = my_out

        # A running list to accumulate filtered answers across retrieval
        # strategies.  This will be used both for early stopping checks and
        # final aggregation.
        combined_filtered_answers: List[Any] = []

        # ------------------------------------------------------------------
        # Tavily Web Search
        # ------------------------------------------------------------------
        if st.session_state.tavily_selected:
            try:
                _check_stop_search()
                # Tavily retreiver
                st.session_state.Tavily_retriever = CustomRetriever_Tavily(
                    vectorstore=None, search_kwargs={"k": configs['retrieval']['max_return_docs']})   

                # Only perform the Tavily search if the strict logbook flag
                # indicates that web search is allowed.  Otherwise skip to
                # subsequent retrieval methods.
                if contains_word(strict_logbook_FLAG, "False"):
                    with traced_spinner("Fetching Documents via Web Search"):
                        # Remove double quotes to avoid quoting issues in the
                        # underlying search API.  This preserves the original
                        # sanitisation logic.
                        safe_query = user_query.replace('"', " ")
                        docs_Tavily = (
                            st.session_state.Tavily_retriever.get_relevant_documents(
                                safe_query
                            )
                        )
                    print_wide_line_message("Tavily WebSearch Retrieval Results")
                    pretty_print_docs(docs_Tavily)
                    with traced_spinner(
                        f"Verifying {len(docs_Tavily)} Web‑Search retrieved documents"
                    ):
                        if st.session_state.useGroq:
                            (
                                filtered_retrieved_docs_Tavily,
                                filtered_retrieved_docs_LLM_answers_Tavily,
                            ) = verify_and_filter_retrieved_docs_v2_parallel(
                                user_query, docs_Tavily
                            )
                        else:
                            (
                                filtered_retrieved_docs_Tavily,
                                filtered_retrieved_docs_LLM_answers_Tavily,
                            ) = verify_and_filter_retrieved_docs_v2(user_query, docs_Tavily)

                    # Update the combined answers and perform early stopping
                    # checks.  We extend rather than reassign to preserve
                    # references to previous results.
                    combined_filtered_answers.extend(
                        filtered_retrieved_docs_LLM_answers_Tavily
                    )
                    trace_add(
                        "Web search (Tavily) results",
                        detail=f"Fetched {len(docs_Tavily)} docs; verified {len(filtered_retrieved_docs_LLM_answers_Tavily)} relevant passages."
                    )
                    if (
                        st.session_state.early_stop_enabled
                        and len(combined_filtered_answers)
                        >= configs["retrieval"]["MIN_VERIFIED_DOCS"]
                    ):
                        # Build a tentative answer using the combined
                        # documents and evaluate its relevance.  If the
                        # relevance is above threshold we return early.
                        (
                            FINAL_RESPONSE,
                            QA_ANSWER,
                            WIKI_SUMMARY,
                            MORE_DETAILS,
                            _filtered,
                            QA_ANSWER_AGGREGATOR,
                        ) = ensembele_superposition_answer_from_docs(
                            combined_filtered_answers, user_query
                        )
                        BASELINE_ANSWER = QA_ANSWER + "\n\n" + WIKI_SUMMARY
                        RELEVANCE_SCORE = get_search_specific_relevance_score(
                            user_query,
                            BASELINE_ANSWER,
                            num_repeats=configs["retrieval"][
                                "RELEVANCE_SCORE_THRESHOLD_CHECK_N_TIMES"
                            ],
                        )
                        if (
                            RELEVANCE_SCORE
                            > configs["retrieval"]["RELEVANCE_SCORE_THRESHOLD"]
                        ):
                            trace_add("Early stopping", detail=f"Verified passages={len(combined_filtered_answers)}; relevance_score={RELEVANCE_SCORE:.2f}")
                            return(
                                FINAL_RESPONSE,
                                QA_ANSWER,
                                WIKI_SUMMARY,
                                MORE_DETAILS,
                                combined_filtered_answers,
                                QA_ANSWER_AGGREGATOR,
                            )
            except Exception as myEXP:
                # On error, record the failure and continue with other methods.
                print("Error with Tavily-WebSearch Search ")
                print(myEXP)
                docs_Tavily = []
                filtered_retrieved_docs_Tavily = []
                filtered_retrieved_docs_LLM_answers_Tavily = []

        # ------------------------------------------------------------------
        # FAISS Semantic Search
        # ------------------------------------------------------------------
        if st.session_state.faiss_selected:

            try:
                _check_stop_search()                
                ##########################
                # First load all retrievers    
                st.session_state.bm25_retriever, st.session_state.retriever, st.session_state.qa_chain, st.session_state.FAISS_retriever,st.session_state.DuckDuckGo_retriever,st.session_state.Tavily_retriever  = get_best_VecDB_info(VecDB_TYPE_ACTUAL,
                                                                                                                                                            st,
                                                                                                                                                            st.session_state.max_return_docs,
                                                                                                                                                            configs['retrieval'][
                                                                                                                                                                'ensemble_retriever_bm25_relative_weight'],
                                                                                                                                                            configs['retrieval'][
                                                                                                                                                                'ensemble_retriever_FAISS_relative_weight'],
                                                                                                                                                            configs['retrieval'][
                                                                                                                                                                'ensemble_retriever_DuckDuckGo_relative_weight'],
                                                                                                                                                            configs['generate'][
                                                                                                                                                                'enable_context_llm_filtering'],
                                                                                                                                                            user_input)
                
                # create FAISS vectorstore
                if 'vectorstore' not in st.session_state:
                    if os.path.isdir(faiss_persist_directory):
                        print(f"FAISS vectorstore exists at {faiss_persist_directory}")
                        st.session_state.vectorstore = load_faiss_vectorstore(
                            faiss_persist_directory)
                        print(f"FAISS vectorstore loaded.")
                    else:
                        BASE_RAG_PATH = "./RAG_DataSets"
                        datasets_config = configs['data']['datasets']
                        RAG_DataSet_directory = f"./RAG_DataSets/{configs['data']['datasets']}/"
                        if os.path.normpath(faiss_persist_directory) != os.path.normpath("./faiss/ALL_V2"):
                            print(f"{faiss_persist_directory} Directory does not exist. Creating a new one")
                            print(f"RAG Data directory:{RAG_DataSet_directory} ")
                            common_params = {
                                'chunk_size': configs['retrieval']['chunk_size'],
                                'chunk_overlap': configs['retrieval']['chunk_overlap']
                            }

                            if os.path.normpath(RAG_DataSet_directory) == os.path.normpath("./RAG_DataSets/ArxivData/"):
                                all_splits_data = fetch_arxiv_abstracts(
                                    keyword=configs['retrieval']['arxiv_keyword'],
                                    start_date_str=configs['retrieval']['arxiv_start_date_str'],
                                    end_date_str=datetime.now().strftime(
                                        '%Y-%m-%d') if configs['retrieval']['arxiv_start_date_str'] == "now" else configs['retrieval']['arxiv_start_date_str'],
                                    max_results_per_year=configs['retrieval']['arxiv_max_results_per_year'],
                                    max_try=configs['retrieval']['arxiv_max_try']
                                )
                                
                            elif os.path.normpath(RAG_DataSet_directory) == os.path.normpath("./RAG_DataSets/ALL_V2/"):
                                arxiv = fetch_arxiv_abstracts(
                                    keyword=configs['retrieval']['arxiv_keyword'],
                                    start_date_str=configs['retrieval']['arxiv_start_date_str'],
                                    end_date_str=datetime.now().strftime('%Y-%m-%d') if configs['retrieval']['arxiv_start_date_str'] == "now" else configs['retrieval']['arxiv_start_date_str'],
                                    max_results_per_year=configs['retrieval']['arxiv_max_results_per_year'],
                                    max_try=configs['retrieval']['arxiv_max_try']
                                )
                                latex = extract_latex_data(f"{BASE_RAG_PATH}/LatexData/", **common_params)
                                text = extract_text_data(f"{BASE_RAG_PATH}/TextData/", **common_params)
                                pdf = extract_pdf_data_V3(f"{BASE_RAG_PATH}/DocPDF/", extraction_method=configs['data']['pdf_extraction_method'], **common_params)
                                jsonl = extract_JSONL_data(f"{BASE_RAG_PATH}/JSONLData/", chunk_size=common_params['chunk_size'])
                                
                                all_splits_data = arxiv + pdf + latex + jsonl

                            else:
                                handler_map = {
                                    "DocPDF": lambda path: extract_pdf_data_V3(path, extraction_method=configs['data']['pdf_extraction_method'], **common_params),
                                    "TextData": lambda path: extract_text_data(path, **common_params),
                                    "AudioTextData": lambda path: extract_text_data(path, **common_params),
                                    "LatexData": lambda path: extract_latex_data(path, **common_params),
                                    "JSONLData": lambda path: extract_JSONL_data(path, chunk_size=common_params['chunk_size']),
                                    "LogbookData": lambda path: extract_heyligo_ProFreports_csv_data(path, chunk_size=common_params['chunk_size']),
                                }

                                if datasets_config in handler_map:
                                    all_splits_data = handler_map[datasets_config](RAG_DataSet_directory)
                                else:
                                    all_splits_data = []

                            all_splits = all_splits_data
                            random.shuffle(all_splits)
                            if configs['retrieval']['enable_limit_max_embed_docs']:
                                print(
                                    f'ENABLE_LIMIT_MAX_EMBED_DOCS set to 1, Only embedding the first {configs["retrieval"]["max_embed_docs"]} documents')
                                all_splits = all_splits[0:configs['retrieval']
                                                        ['max_embed_docs']]
                            if configs['retrieval']['enable_text_preprocess']:
                                print("ENABLE_TEXT_PREPROCESS is enabled.")
                                preprocessed_documents = [
                                    preprocess_page_content_text(doc) for doc in all_splits]
                            else:
                                preprocessed_documents = all_splits
                            print("Embedding documents to vectorstore")
                            batch_size = 20
                            total_docs = len(preprocessed_documents)
                            for i in tqdm(range(0, total_docs, batch_size), desc="Processing documents"):
                                batch_docs = preprocessed_documents[i:i + batch_size]
                                if 'vectorstore' not in st.session_state:
                                    st.session_state.vectorstore = FAISS.from_documents(
                                        batch_docs, faiss_embeddings)
                                else:
                                    new_vectorstore = FAISS.from_documents(
                                        batch_docs, faiss_embeddings)
                                    st.session_state.vectorstore.merge_from(new_vectorstore)
                        else:
                            dbs = []
                            for vectorstore_option in configs['data']['faiss_vector_store_merges']:
                                if vectorstore_option != "ALL_V2":
                                    db = load_faiss_vectorstore(
                                        f"./faiss/{vectorstore_option}")
                                    dbs.append(db)
                            st.session_state.vectorstore = dbs[0]
                            if dbs:
                                for db in dbs[1:]:
                                    st.session_state.vectorstore.merge_from(db)
                            else:
                                st.error("No vector stores were loaded.")

                        # persist vectorstore
                        save_faiss_vectorstore(
                            st.session_state.vectorstore, faiss_persist_directory)

                # load docs from vectorstore (for use in BM25+)
                if 'vectorstore_docs' not in st.session_state:
                    st.session_state.vectorstore_docs = get_docs_from_faiss_vectorstore(
                        st.session_state.vectorstore)

                # guardrail if initial_retrieval_count > len(vectorstore)
                max_num_docs_vectorstore = len(st.session_state.vectorstore_docs)
                if st.session_state.initial_retrieval_count > max_num_docs_vectorstore:
                    st.session_state.initial_retrieval_count = max_num_docs_vectorstore
                    
                if st.session_state.max_return_docs > st.session_state.initial_retrieval_count:
                    st.session_state.max_return_docs = st.session_state.initial_retrieval_count


                # Initialie BM25 Retriever
                st.session_state.bm25_retriever = CustomRetriever_BM25(
                    vectorstore=st.session_state.vectorstore, search_kwargs={"k": st.session_state.max_return_docs})


                # MODIFIED CHROMA retriever
                # st.session_state.chroma_retriever = CustomRetriever(vectorstore=st.session_state.vectorstore,search_type="mmr",search_kwargs={"k": max_return_docs})
                if configs['retrieval']['disable_llm_embed_retreiver']:
                    st.session_state.chroma_retriever = st.session_state.bm25_retriever

                # Initialie FAISS Retriever
                st.session_state.FAISS_retriever = CustomRetriever_FAISS(
                    vectorstore=st.session_state.vectorstore, search_kwargs={"k": st.session_state.max_return_docs})



                # initialize the ensemble retriever
                st.session_state.retriever = EnsembleRetriever(
                    retrievers=[st.session_state.bm25_retriever, st.session_state.FAISS_retriever], weights=[
                        configs['retrieval']['ensemble_retriever_bm25_relative_weight'], configs['retrieval']['ensemble_retriever_FAISS_relative_weight']]
                )


                # Update EnsembleRetriever
                # Retriver with Contextual compression
                if configs['generate']['enable_context_llm_filtering'] == 1:
                    compressor = LLMChainFilter.from_llm(
                        st.session_state.llm, custom_llm_chain_filter_prompt_template)
                    st.session_state.retriever = ContextualCompressionRetriever(
                        base_compressor=compressor, base_retriever=st.session_state.retriever
                    )


                if 'qa_chain' not in st.session_state:
                    st.session_state.qa_chain = RetrievalQA.from_chain_type(
                        llm=st.session_state.llm,
                        chain_type='stuff',
                        retriever=st.session_state.retriever,
                        verbose=configs['retrieval']['enable_verbose'],
                        chain_type_kwargs={
                            "verbose": configs['retrieval']['enable_verbose'],
                            "prompt": st.session_state.prompt,
                            "memory": st.session_state.memory,
                        }
                    )
            except Exception as myEXP:
                # On error, record the failure and continue with other methods.
                print("Error with FAISS Vectorstore loading ")
                print(myEXP)
                ##########################



            try:
                _check_stop_search()
                if contains_word(strict_logbook_FLAG, "False"):
                    with traced_spinner("Fetching Documents via Semantic Search"):
                        docs_FAISS = st.session_state.FAISS_retriever.invoke(
                            user_query
                        )
                    print_wide_line_message("FAISS Semantic Retrieval Results")
                    pretty_print_docs(docs_FAISS)
                    with traced_spinner(
                        f"Verifying {len(docs_FAISS)} Semantic‑Search retrieved documents"
                    ):
                        if st.session_state.useGroq:
                            (
                                filtered_retrieved_docs_FAISS,
                                filtered_retrieved_docs_LLM_answers_FAISS,
                            ) = verify_and_filter_retrieved_docs_v2_parallel(
                                user_query, docs_FAISS
                            )
                        else:
                            (
                                filtered_retrieved_docs_FAISS,
                                filtered_retrieved_docs_LLM_answers_FAISS,
                            ) = verify_and_filter_retrieved_docs_v2(
                                user_query, docs_FAISS
                            )
                    # Extend combined answers
                    combined_filtered_answers.extend(
                        filtered_retrieved_docs_LLM_answers_FAISS
                    )
                    trace_add(
                        "Semantic search (FAISS) results",
                        detail=f"Fetched {len(docs_FAISS)} docs; verified {len(filtered_retrieved_docs_LLM_answers_FAISS)} relevant passages."
                    )
                    if (
                        st.session_state.early_stop_enabled
                        and len(combined_filtered_answers)
                        >= configs["retrieval"]["MIN_VERIFIED_DOCS"]
                    ):
                        (
                            FINAL_RESPONSE,
                            QA_ANSWER,
                            WIKI_SUMMARY,
                            MORE_DETAILS,
                            _filtered,
                            QA_ANSWER_AGGREGATOR,
                        ) = ensembele_superposition_answer_from_docs(
                            combined_filtered_answers, user_query
                        )
                        BASELINE_ANSWER = QA_ANSWER + "\n\n" + WIKI_SUMMARY
                        RELEVANCE_SCORE = get_search_specific_relevance_score(
                            user_query,
                            BASELINE_ANSWER,
                            num_repeats=configs["retrieval"][
                                "RELEVANCE_SCORE_THRESHOLD_CHECK_N_TIMES"
                            ],
                        )
                        if (
                            RELEVANCE_SCORE
                            > configs["retrieval"]["RELEVANCE_SCORE_THRESHOLD"]
                        ):
                            trace_add("Early stopping", detail=f"Verified passages={len(combined_filtered_answers)}; relevance_score={RELEVANCE_SCORE:.2f}")
                            return(
                                FINAL_RESPONSE,
                                QA_ANSWER,
                                WIKI_SUMMARY,
                                MORE_DETAILS,
                                combined_filtered_answers,
                                QA_ANSWER_AGGREGATOR,
                            )
            except Exception as myEXP:
                print("Error with FAISS Search ")
                print(myEXP)
                docs_FAISS = []
                filtered_retrieved_docs_FAISS = []
                filtered_retrieved_docs_LLM_answers_FAISS = []

        # ------------------------------------------------------------------
        # BM25 Lexical Search
        # ------------------------------------------------------------------
        if st.session_state.bm25_selected:
            try:
                _check_stop_search()
                if contains_word(strict_logbook_FLAG, "False"):
                    # Determine whether lexical search is appropriate via the
                    # LEXICAL_FLAG prompt.  This logic is unchanged from
                    # upstream.
                    my_template = configs["prompts"]["LEXICAL_FLAG_template"]["template"]
                    my_prompt = PromptTemplate(
                        input_variables=["user_query"], template=my_template
                    )
                    my_formatted_prompt = my_prompt.format(user_query=user_query)
                    my_out = st.session_state.llm(my_formatted_prompt)
                    lexical_FLAG = my_out
                    if contains_word(lexical_FLAG, "True"):
                        with traced_spinner("Fetching Documents via Lexical Search"):
                            docs_BM25 = st.session_state.bm25_retriever.get_relevant_documents(
                                user_query
                            )
                    else:
                        docs_BM25 = []
                    with st.spinner(
                        f"Verifying {len(docs_BM25)} Lexical‑Search retrieved documents"
                    ):
                        # Both branches currently call the same v2 function; this
                        # mirrors the original code despite the duplicate check.
                        if st.session_state.useGroq:
                            (
                                filtered_retrieved_docs_BM25,
                                filtered_retrieved_docs_LLM_answers_BM25,
                            ) = verify_and_filter_retrieved_docs_v2(
                                user_query, docs_BM25
                            )
                        else:
                            (
                                filtered_retrieved_docs_BM25,
                                filtered_retrieved_docs_LLM_answers_BM25,
                            ) = verify_and_filter_retrieved_docs_v2(
                                user_query, docs_BM25
                            )
                    combined_filtered_answers.extend(
                        filtered_retrieved_docs_LLM_answers_BM25
                    )
                    trace_add(
                        "Lexical search (BM25) results",
                        detail=f"Fetched {len(docs_BM25)} docs; verified {len(filtered_retrieved_docs_LLM_answers_BM25)} relevant passages."
                    )
                    if (
                        st.session_state.early_stop_enabled
                        and len(combined_filtered_answers)
                        >= configs["retrieval"]["MIN_VERIFIED_DOCS"]
                    ):
                        (
                            FINAL_RESPONSE,
                            QA_ANSWER,
                            WIKI_SUMMARY,
                            MORE_DETAILS,
                            _filtered,
                            QA_ANSWER_AGGREGATOR,
                        ) = ensembele_superposition_answer_from_docs(
                            combined_filtered_answers, user_query
                        )
                        BASELINE_ANSWER = QA_ANSWER + "\n\n" + WIKI_SUMMARY
                        RELEVANCE_SCORE = get_search_specific_relevance_score(
                            user_query,
                            BASELINE_ANSWER,
                            num_repeats=configs["retrieval"][
                                "RELEVANCE_SCORE_THRESHOLD_CHECK_N_TIMES"
                            ],
                        )
                        if (
                            RELEVANCE_SCORE
                            > configs["retrieval"]["RELEVANCE_SCORE_THRESHOLD"]
                        ):
                            trace_add("Early stopping", detail=f"Verified passages={len(combined_filtered_answers)}; relevance_score={RELEVANCE_SCORE:.2f}")
                            return(
                                FINAL_RESPONSE,
                                QA_ANSWER,
                                WIKI_SUMMARY,
                                MORE_DETAILS,
                                combined_filtered_answers,
                                QA_ANSWER_AGGREGATOR,
                            )
                else:
                    docs_BM25 = []
            except Exception as myEXP:
                print("Error with BM25+ Search ")
                print(myEXP)
                docs_BM25 = []
                filtered_retrieved_docs_BM25 = []
                filtered_retrieved_docs_LLM_answers_BM25 = []

        # ------------------------------------------------------------------
        # HeyLIGO Search
        # ------------------------------------------------------------------
        if st.session_state.heyligo_selected:
            try:
                _check_stop_search()
                # HEYLIGO uses its own flag to decide whether to perform a
                # search.  This remains unchanged from the original
                # implementation.  We override only if the user has disabled
                # this search via the checkbox above.
                my_template = configs["prompts"]["HEYLIGO_FLAG_template"]["template"]
                my_prompt = PromptTemplate(
                    input_variables=["user_query"], template=my_template
                )
                my_formatted_prompt = my_prompt.format(user_query=user_query)
                my_out = st.session_state.llm(my_formatted_prompt)
                heyligo_FLAG = my_out
                if contains_word(heyligo_FLAG, "True"):
                    with traced_spinner("Fetching Documents via HeyLIGO Search"):

                        try:
                            docs_HeyLIGO = []
                            docs_HeyLIGO, text_unused = heyligo_selenium_search( query=user_query,
                                                                           sources=["LLO", "LHO","DCC"],
                                                                        fetch_content=True,
                                                                         max_results={"LLO": 30,"LHO":30, "DCC": 30},
                                                                         verbose=True)
                            # print each document's metadata and a snippet of its content
                            for i, doc in enumerate(docs_HeyLIGO, start=1):
                                print(f"Document {i}")
                                print("Metadata:", doc.metadata)
                                # show first 300 characters of the content; adjust as needed
                                print("Content snippet:\n", doc.page_content[:300], "...\n")

                        except:
                            with traced_spinner("Fetching Documents via HeyLIGO backup Search"):
                                import re
                                from typing import List  # noqa: F401

                                heyligo_faiss_persist_directory = "./faiss/LogbookData"
                                # Map section -> correct aLOG site
                                _LOGBOOK_SITE = {
                                    "l1": "https://alog.ligo-la.caltech.edu/aLOG/index.php?callRep=",
                                    "h1": "https://alog.ligo-wa.caltech.edu/aLOG/index.php?callRep=",
                                }
                                _RE_ENTRY_NUM = re.compile(r"(?i)\blogbook\s*entry\s*:\s*(\d+)\b")
                                _RE_ENTRY_URL = re.compile(r"(?i)\blogbook\s*entry\s*:\s*https?://\S+")
                                _RE_SECTION_TX = re.compile(
                                    r"(?i)\bsection\s*:\s*([lh]1)\b"
                                )
                                _RE_SECTION_MD = re.compile(r"(?i)\b([lh]1)\b")

                                def _detect_section_from_doc(doc) -> str:
                                    md = dict(getattr(doc, "metadata", {}) or {})
                                    for key in ("section", "ifo", "site", "detector"):
                                        val = md.get(key)
                                        if isinstance(val, str):
                                            m = _RE_SECTION_MD.search(val)
                                            if m:
                                                return m.group(1).lower()
                                    text = getattr(doc, "page_content", "") or ""
                                    m = _RE_SECTION_TX.search(text)
                                    return m.group(1).lower() if m else ""

                                def inject_logbook_urls(
                                    docs: List[Any], enable: bool = True, add_metadata: bool = True
                                ) -> List[Any]:
                                    if not enable:
                                        return docs
                                    out = []
                                    for d in docs:
                                        text = getattr(d, "page_content", "") or ""
                                        if not text or _RE_ENTRY_URL.search(text) is not None:
                                            out.append(d)
                                            continue
                                        m_entry = _RE_ENTRY_NUM.search(text)
                                        if m_entry is None:
                                            out.append(d)
                                            continue
                                        section = _detect_section_from_doc(d)
                                        base = _LOGBOOK_SITE.get(section)
                                        if not base:
                                            out.append(d)
                                            continue
                                        callrep = m_entry.group(1)
                                        url = f"{base}{callrep}"
                                        new_text = _RE_ENTRY_NUM.sub(
                                            f"logbook entry: {url}", text, count=1
                                        )
                                        new_md = dict(getattr(d, "metadata", {}) or {})
                                        if add_metadata:
                                            new_md.setdefault("logbook_url", url)
                                            new_md.setdefault("logbook_callrep", int(callrep))
                                            new_md.setdefault("logbook_site", section.upper())
                                        out.append(
                                            type(d)(page_content=new_text, metadata=new_md)
                                        )
                                    return out

                                if "heyligo_vectorstore" not in st.session_state:
                                    st.session_state.heyligo_vectorstore = load_faiss_vectorstore(
                                        heyligo_faiss_persist_directory
                                    )
                                    st.session_state.heyligo_vectorstore_docs = (
                                        get_docs_from_faiss_vectorstore(
                                            st.session_state.heyligo_vectorstore
                                        )
                                    )
                                    from langchain_community.embeddings.ollama import (
                                        OllamaEmbeddings,
                                    )
                                    emb = st.session_state.heyligo_vectorstore.embedding_function
                                    if isinstance(emb, OllamaEmbeddings) and not hasattr(
                                        emb, "headers"
                                    ):
                                        setattr(emb.__class__, "headers", None)
                                results = get_best_VecDB_info(
                                    "LogbookData",
                                    st,
                                    st.session_state.max_return_docs,
                                    configs["retrieval"]["ensemble_retriever_bm25_relative_weight"],
                                    configs["retrieval"]["ensemble_retriever_FAISS_relative_weight"],
                                    configs["retrieval"]["ensemble_retriever_DuckDuckGo_relative_weight"],
                                    configs["generate"]["enable_context_llm_filtering"],
                                    user_query,
                                )
                                st.session_state.heyligo_FAISS_retriever = results[3]
                                docs_HeyLIGO = st.session_state.heyligo_FAISS_retriever.get_relevant_documents(
                                    user_query
                                )
                                docs_HeyLIGO = inject_logbook_urls(
                                    docs_HeyLIGO,
                                    enable=True,
                                    add_metadata=True,
                                )
                else:
                    docs_HeyLIGO = []
                print_wide_line_message("HeyLIGO Retrieval Results")
                with traced_spinner(
                    f"Verifying {len(docs_HeyLIGO)} HeyLIGO‑Search retrieved documents"
                ):
                    if st.session_state.useGroq:
                        (
                            filtered_retrieved_docs_HeyLIGO,
                            filtered_retrieved_docs_LLM_answers_HeyLIGO,
                        ) = verify_and_filter_retrieved_docs_v2_parallel(
                            user_query, docs_HeyLIGO
                        )
                    else:
                        (
                            filtered_retrieved_docs_HeyLIGO,
                            filtered_retrieved_docs_LLM_answers_HeyLIGO,
                        ) = verify_and_filter_retrieved_docs_v2(
                            user_query, docs_HeyLIGO
                        )
                combined_filtered_answers.extend(
                    filtered_retrieved_docs_LLM_answers_HeyLIGO
                )
                trace_add(
                    "HeyLIGO search results",
                    detail=f"Fetched {len(docs_HeyLIGO)} docs; verified {len(filtered_retrieved_docs_LLM_answers_HeyLIGO)} relevant passages."
                )
                if (
                    st.session_state.early_stop_enabled
                    and len(combined_filtered_answers)
                    >= configs["retrieval"]["MIN_VERIFIED_DOCS"]
                ):
                    (
                        FINAL_RESPONSE,
                        QA_ANSWER,
                        WIKI_SUMMARY,
                        MORE_DETAILS,
                        _filtered,
                        QA_ANSWER_AGGREGATOR,
                    ) = ensembele_superposition_answer_from_docs(
                        combined_filtered_answers, user_query
                    )
                    BASELINE_ANSWER = QA_ANSWER + "\n\n" + WIKI_SUMMARY
                    RELEVANCE_SCORE = get_search_specific_relevance_score(
                        user_query,
                        BASELINE_ANSWER,
                        num_repeats=configs["retrieval"][
                            "RELEVANCE_SCORE_THRESHOLD_CHECK_N_TIMES"
                        ],
                    )
                    if (
                        RELEVANCE_SCORE
                        > configs["retrieval"]["RELEVANCE_SCORE_THRESHOLD"]
                    ):
                        trace_add("Early stopping", detail=f"Verified passages={len(combined_filtered_answers)}; relevance_score={RELEVANCE_SCORE:.2f}")
                        return(
                            FINAL_RESPONSE,
                            QA_ANSWER,
                            WIKI_SUMMARY,
                            MORE_DETAILS,
                            combined_filtered_answers,
                            QA_ANSWER_AGGREGATOR,
                        )
            except Exception as myEXP:
                print("Error with HeyLIGO Search ")
                print(myEXP)
                docs_HeyLIGO = []
                filtered_retrieved_docs_HeyLIGO = []
                filtered_retrieved_docs_LLM_answers_HeyLIGO = []

        # ------------------------------------------------------------------
        # Final aggregation and return
        # ------------------------------------------------------------------
        # After processing all selected retrieval strategies, we aggregate all
        # filtered answers.  If early stopping was disabled and the total
        # number of verified documents is below the configured minimum, we
        # discard them and send an empty list to the answer generator.
        if len(combined_filtered_answers) < configs["retrieval"]["MIN_VERIFIED_DOCS"]:
            combined_filtered_answers = []
        (
            FINAL_RESPONSE,
            QA_ANSWER,
            WIKI_SUMMARY,
            MORE_DETAILS,
            _filtered,
            QA_ANSWER_AGGREGATOR,
        ) = ensembele_superposition_answer_from_docs(
            combined_filtered_answers, user_query
        )
        return (
            FINAL_RESPONSE,
            QA_ANSWER,
            WIKI_SUMMARY,
            MORE_DETAILS,
            combined_filtered_answers,
            QA_ANSWER_AGGREGATOR,
        )



# Iterative Summarization using LangGraph
async def run_langgraph_summarization(llm,docs,user_query):
     # Run the summarization
    langgraph_summary = await langgraph_summarize_documents(llm, docs,user_query)
    return langgraph_summary


def convert_all_latex_delimiters(text):
    """
    Convert LaTeX equations enclosed within \[ ... \] or [ ... ] to $$ ... $$,
    ensuring that only brackets containing LaTeX commands are replaced.
    Args:
        text (str): The input string containing LaTeX equations.
    Returns:
        str: The modified string with LaTeX equations enclosed within $$ ... $$.
    """
    # Enhanced regex pattern to match \[ ... \] or [ ... ] only if content contains backslashes
    pattern = r'\\\[(.*?\\.*?)\\\]|\[(.*?\\.*?)\]'

    def replacer(match):
        content = match.group(1) if match.group(
            1) is not None else match.group(2)
        return f'$$ {content} $$'
    converted_text = re.sub(pattern, replacer, text, flags=re.DOTALL)
    return converted_text


def streamlit_markdown_write(response_result):
    # nicely_format_latex_eqns
    response_result = nicely_format_latex_eqns(response_result)
    full_response = ""
    for chunk in response_result.split():
        full_response += chunk + " "
        time.sleep(0.05)
        # Add a blinking cursor to simulate typing
        message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)
    message_placeholder.markdown(response_result)


def fetch_system1_OUTPUT(user_query):

    context = get_chat_history_string()
    
    # use reflection_prompt_system1_answer
    llm1_out = reflection_prompt_system1_answer(
        user_query, myLLM=st.session_state.llm,context=context)
    
    llm2_out = reflection_prompt_system1_answer(
        user_query, myLLM=st.session_state.llm_2,context=context)
    
    print_wide_line_message("LLM1 System-1 Response")
    print(llm1_out)
    print_wide_line_message("LLM2 System-1 Response")
    print(llm2_out)    


    # Use SYSTEM1_ANSWER_CONSISTENCY_CHECKER_template
    my_template = configs['prompts']['SYSTEM1_ANSWER_CONSISTENCY_CHECKER_template']['template']
    my_prompt = PromptTemplate(input_variables=["user_query,""llm1_out","llm2_out"],template=my_template)
    my_formatted_prompt = my_prompt.format(user_query=user_query,llm1_out=llm1_out,llm2_out=llm2_out)
    my_out = st.session_state.llm(my_formatted_prompt)

    #breakpoint()

    COMPARE_OUTPUTS = my_out
    CONSISTENT_FLAG = not ("inconsistent" in COMPARE_OUTPUTS.lower())
    SYSTEM1_COMBINED_ANSWER = ""

    if CONSISTENT_FLAG:

        # Use SYSTEM1_ANSWER_AGGREGATOR_template
        my_template = configs['prompts']['SYSTEM1_ANSWER_AGGREGATOR_template']['template']
        my_prompt = PromptTemplate(input_variables=["llm1_out","llm2_out"],template=my_template)
        my_formatted_prompt = my_prompt.format(llm1_out=llm1_out,llm2_out=llm2_out)
        my_out = st.session_state.llm(my_formatted_prompt)

        aggegated_output = my_out


        if 'ANSWER' in aggegated_output:
            # Regular expression to find text between <ANSWER> and </ANSWER>
            pattern = r'<ANSWER>(.*?)</ANSWER>'
            # Find all matches
            SYSTEM1_COMBINED_ANSWER = re.findall(
                pattern, aggegated_output, re.DOTALL)
            SYSTEM1_COMBINED_ANSWER = " \n".join(SYSTEM1_COMBINED_ANSWER)
        else:
            SYSTEM1_COMBINED_ANSWER = aggegated_output



        if len(str(SYSTEM1_COMBINED_ANSWER.strip()))==0:
            print("Check SYSTEM1_ANSWER_AGGREGATOR_template, <ANSWER> tags are not working properly. Showing the full response.")
            SYSTEM1_COMBINED_ANSWER = aggegated_output
            #breakpoint() 

        # Use a LLM-JUDGE to judge the quality of the response
        with traced_spinner("Judging the quality of System1 response"):
            judge_score = GET_LLM_JUDGE_SCORE(
                user_query, SYSTEM1_COMBINED_ANSWER, num_trials=3)
            try:
                if judge_score < 7:
                    CONSISTENT_FLAG = False
            except:
                pass
        print("CONSISTENT_FLAG-0")        
        print(CONSISTENT_FLAG)
    return CONSISTENT_FLAG, SYSTEM1_COMBINED_ANSWER




def GET_LLM_JUDGE_SCORE(user_query, generated_answer, num_trials=3):

    sum_out = []
    for trial in range(num_trials):
        
        # Use LLM_JUDGE_template
        my_template = configs['prompts']['LLM_JUDGE_template']['template']
        my_prompt = PromptTemplate(input_variables=["user_query","generated_answer"],template=my_template)
        my_formatted_prompt = my_prompt.format(user_query=user_query,generated_answer=generated_answer)
        my_out = st.session_state.llm(my_formatted_prompt)

        judge_output = my_out
        print("judge_output")
        print(judge_output)
        pattern = r'<OUTPUT>(.*?)</OUTPUT>'
        # Find all matches
        judge_score = re.findall(pattern, judge_output, re.DOTALL)
        print("judge_score")
        print(judge_score)
        if len(judge_score) == 0:
            judge_score = re.findall(r'\d', judge_output)
        judge_score = judge_score[0]
        print(f"trial:{trial} judge_score:{judge_score}")
        judge_score = int(judge_score)
        sum_out.append(judge_score)
    judge_score = np.sum(sum_out)/len(sum_out)
    return judge_score


def reflection_prompt_system1_answer(user_query, myLLM, context=""):


    # Use SYSTEM1_REFLECTION_template
    my_template = configs['prompts']['SYSTEM1_REFLECTION_template']['template']
    my_prompt = PromptTemplate(input_variables=["user_query","context"],template=my_template)
    my_formatted_prompt = my_prompt.format(user_query=user_query,context=context)
    my_out = myLLM(my_formatted_prompt)


    REFLECTION_OUTPUT = my_out

    # Regular expression to find text between <ANSWER> and </ANSWER>
    pattern = r'<ANSWER>(.*?)</ANSWER>'
    # Find all matches
    QA_ANSWER = re.findall(pattern, REFLECTION_OUTPUT, re.DOTALL)
    QA_ANSWER = " \n".join(QA_ANSWER)
    return QA_ANSWER

# === Minimal Post-Processing Gate (added) ===


def _pp_parse_json_obj(txt: str) -> dict:
    try:
        return json.loads(txt)
    except Exception:
        s, e = txt.find("{"), txt.rfind("}")
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(txt[s:e+1])
            except Exception:
                return {}
    return {}

def maybe_postprocess_turn(user_input: str) -> bool:
    """
    If user asks to post-process the previous assistant output, transform & short-circuit.
    Returns True if it handled the turn (i.e., normal RAG must not run).
    """
    prev_significant = st.session_state.get("_pp_prev_significant_answer")
    if not prev_significant:
        return False  # nothing to post-process yet

    llm = getattr(st.session_state, "llm", None)  # IMPORTANT: raw, stateless generator
    if not callable(llm):
        return False

    prompt = f"""
You are an INTENT+TRANSFORM tool for a chat app.

Decide whether the user wants to POSTPROCESS the PREVIOUS assistant output,
based on these HINT phrases (non-exhaustive examples) and the user message:

HINTS = {PP_HINTS}

Rules:
- If the user asks to reformat/summarize/extract/convert the PREVIOUS_OUTPUT (without asking for new info), intent = "POSTPROCESS".
- If the user asks a NEW question or requests new retrieval, intent = "QUERY".
- If POSTPROCESS, produce the transformed text STRICTLY from PREVIOUS_OUTPUT. Do NOT invent new facts or use outside knowledge.
- Keep equations/LaTeX as-is if requested.
- If "make a table" or "tabulate", return a Markdown table.
- Return JSON ONLY, no prose, in this exact shape:
  {{"intent": "POSTPROCESS" | "QUERY", "output": "<string or empty>"}}

USER_MESSAGE:
{user_input}

PREVIOUS_OUTPUT:
{prev_significant}
""".strip()

    raw = llm(prompt)  # ONE safe call; do not use router-linked helpers
    data = _pp_parse_json_obj(raw if isinstance(raw, str) else str(raw))
    intent = (data.get("intent") or "").upper()
    if intent == "POSTPROCESS":
        trace_add("Post-processing previous answer", detail=f"Instruction: {user_input}")
        raw_out = _pp_output_to_text(data.get("output"))
        out = raw_out.strip() or prev_significant
        trace_add("Post-processing complete", detail=f"Output length: {len(out)} characters")
        st.session_state["_pp_prev_answer"] = out
        # display via your existing helper if available; else fallback
        try:
            streamlit_add_msg(st=st, role="assistant", message=out, persist=True)  # type: ignore[name-defined]
        except Exception:
            with st.chat_message("assistant"):
                st.markdown(out)
        # Persist the assistant post-processed output to the database
        if st.session_state.get("current_conv_id") is not None:
            save_message(st.session_state.current_conv_id, "assistant", out)

        # Update summary memory so follow-ups work reliably
        try:
            maybe_save_turn_to_memory(user_input, out)
        except Exception:
            pass

        # Reset rewrite state so we do not remain stuck in confirm loops
        try:
            _reset_query_rewrite_state()
        except Exception:
            st.session_state.stage = "initial"
            st.session_state.updated_query_for_confirm = None

        st.session_state.refresh_selected_conv = True

        # Attach trace to this assistant message (persists on reruns)
        try:
            _trace_items = copy.deepcopy(st.session_state.get("reasoning_trace", []))
            attach_trace_to_last_assistant_message(_trace_items)
        except Exception:
            pass

        st.stop()  # short-circuit BEFORE any routing/RAG
        return True

    return False
# === End Minimal Post-Processing Gate (added) ===





# ----------------------------------------------------------------------------
# Search-directive guard: avoid running full RAG when user says things like
# 'can you do a search?' without specifying what to search for.
# ----------------------------------------------------------------------------
def _count_content_tokens_for_search_guard(text: str) -> int:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-']*", (text or "").lower())
    stop = {
        'a','an','the','and','or','but','if','then','else','to','of','in','on','for','about','from',
        'please','pls','can','could','would','you','me','we','us','do','does','did','perform','run','execute',
        'search','lookup','look','up','find','browse','google','more','again',
        'it','this','that','those','these','them','something','anything','stuff','info','information','topic','things'
    }
    return sum(1 for t in tokens if t not in stop)

def is_vague_search_directive(user_input: str) -> bool:
    """Return True if the user is asking us to 'do a search' but didn't provide a usable target."""
    q = (user_input or "").strip()
    if not q:
        return False
    ql = q.lower().strip()

    # Detect directive-like search requests (as opposed to questions ABOUT searching).
    directive = False
    if re.match(r"^(can|could|would)\s+(you|u)\s+.*\b(search|look\s+up|lookup|find|browse|google)\b", ql):
        directive = True
    if re.match(r"^(please\s+)?(do|perform|run|execute)\s+(a\s+)?search\b", ql):
        directive = True
    if re.match(r"^(please\s+)?search\b", ql):
        directive = True
    if re.match(r"^(let's|lets)\s+search\b", ql):
        directive = True
    if re.match(r"^(look\s+up|lookup|find|browse|google)\b", ql):
        directive = True

    if not directive:
        return False

    # Exclude content questions like 'search algorithm', 'search space', etc.
    if re.match(r"^search\s+(algorithm|algorithms|space|method|methods|strategy|strategies|technique|techniques)\b", ql):
        return False

    # Extract remainder after the directive
    rem = ql
    rem = re.sub(r"^(can|could|would)\s+(you|u)\s+", "", rem)
    rem = re.sub(r"^please\s+", "", rem)
    rem = re.sub(r"^(do|perform|run|execute)\s+(a\s+)?", "", rem)
    rem = re.sub(r"^(let's|lets)\s+", "", rem)
    rem = re.sub(r"^(search|look\s+up|lookup|find|browse|google)\b", "", rem).strip()
    rem = re.sub(r"^(for|about|on)\b", "", rem).strip()

    # Remove common filler words (keeps proper nouns / technical terms)
    rem2 = re.sub(r"\b(more|something|anything|stuff|info|information|topic|things)\b", "", rem).strip()

    if not rem2:
        return True
    if rem2 in {"it", "this", "that", "those", "these", "them"}:
        return True
    if _count_content_tokens_for_search_guard(rem2) < 2:
        return True

    return False

def maybe_clarify_search_request_turn(user_input: str) -> bool:
    """If user asks to 'do a search' with no clear target, ask a clarifying question and stop."""
    if not is_vague_search_directive(user_input):
        return False

    msg = (
        "I can search, but I need a specific target.\n\n"
        "What should I search for—topic, keywords, a paper title, an author, or a concrete question?\n"
        "Examples:\n"
        "- `Search for papers by Albert Einstein on general theory of relativity`\n"
        "- `Summarize the main results of the \"Attention is all you need\" paper`\n"
        "- `Find tidal deformability constraints for neutron stars`"
    )

    trace_add("Clarification requested", detail="User asked to search without specifying what to search for.")

    try:
        streamlit_add_msg(st=st, role="assistant", message=msg, persist=True)  # type: ignore[name-defined]
    except Exception:
        with st.chat_message("assistant"):
            st.markdown(msg)

    if st.session_state.get("current_conv_id") is not None:
        save_message(st.session_state.current_conv_id, "assistant", msg)

    try:
        maybe_save_turn_to_memory(user_input, msg)
    except Exception:
        pass

    st.session_state["_pp_prev_answer"] = msg
    st.session_state.refresh_selected_conv = True

    try:
        _reset_query_rewrite_state()
    except Exception:
        st.session_state.stage = "initial"
        st.session_state.updated_query_for_confirm = None

    # Attach trace to this assistant message (persists on reruns)
    try:
        _trace_items = copy.deepcopy(st.session_state.get("reasoning_trace", []))
        attach_trace_to_last_assistant_message(_trace_items)
    except Exception:
        pass

    st.stop()
    return True






# Chat input
if user_input := st.chat_input("You:", key="user_input"):

    # New user message should not inherit an old stop request
    st.session_state.stop_search_requested = False

    # Append user's message to the in‑memory chat history
    user_message = {"role": "user", "message": user_input}
    st.session_state.chat_history.append(user_message)

    # Save the user's message to the database (if a conversation is selected)
    if st.session_state.get("current_conv_id") is not None:
        save_message(st.session_state.current_conv_id, "user", user_input)

    with st.chat_message("user"):
        st.markdown(user_input)

    # Reset per-turn reasoning trace
    trace_reset(user_input)
    trace_add("Received user message")

    # ---------------------------------------------------------
    # IMPORTANT: run any short-circuit handlers BEFORE opening the
    # assistant spinner context. If a handler uses st.stop(), calling
    # it inside a spinner can leave the spinner stuck on screen.
    # ---------------------------------------------------------
    if is_empty_or_whitespace(user_input) != 1:
        user_input = (user_input or "").strip()
        user_input = user_input.lower()
        print(f"\n Initial USER_INPUT: \n {user_input}")

        # Short-circuit: post-process previous assistant output (summarize, bullets, table, etc.)
        print_wide_line_message("Before maybe_postprocess_turn(user_input)") 
        _pp_debug_print_memory_state()        
        if maybe_postprocess_turn(user_input):
            pass  # st.stop() executed inside

        # Short-circuit: vague 'do a search' requests -> ask for clarification
        if maybe_clarify_search_request_turn(user_input):
            pass  # st.stop() executed inside

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
                
            with traced_spinner("💡 Assistant is thinking"):
                message_placeholder = st.empty()
                if is_empty_or_whitespace(user_input) != 1:

                    if 'stage' not in st.session_state:
                        st.session_state.stage = 'initial'

                    concatenated_definitions = ""
                    if configs['general']['enable_llm_query_improvement'] == True:
                        RAG_QUERY, concatenated_definitions = process_user_query_and_switch_states()
                    else:
                        RAG_QUERY = user_input

                    print(f'\n RAG_QUERY: \n {RAG_QUERY}')
                    print(f'\n concatenated_definitions: \n {concatenated_definitions}')

                    # Initialize an empty dictionary
                    response = {}  # Initialize an empty dictionary
                    # Set the 'result' key to an empty string
                    response['result'] = ""

                    if RAG_QUERY is not None:

                        # get FINAL_RAG_QUERY method:2
                        pattern = r"Did you mean to ask: (.+)"
                        match = re.search(pattern, RAG_QUERY)
                        if match:
                            FINAL_RAG_QUERY = match.group(1)
                        else:
                            FINAL_RAG_QUERY = RAG_QUERY

                        sensible_question_final_check_FLAG = st.session_state.llm_query_improve(f'''
                                                                                    Return <NO> if the <TEXT> contains no significant question or has just some pleasantries. 
                                                                                    Return <YES> if the <TEXT> is asking to verify the correctness of the paragraph.
                                                                                    If it contains significant and technical statement or question only then return <YES>. 
                                                                                    The question should have a subject or object. Only return either <YES> or <NO>.
                                                                                    Here is the <TEXT>: {FINAL_RAG_QUERY}
                                                                                    ''')

                        if contains_word(sensible_question_final_check_FLAG, "NO"):
                            llm_response = "Let me know what you would like to know."
                            st.markdown(llm_response)
                            st.session_state.chat_history.append({"role": "assistant", "message": llm_response})

                            # Persist the assistant's response to the database
                            if st.session_state.get("current_conv_id") is not None:
                                save_message(st.session_state.current_conv_id, "assistant", llm_response)

                            st.rerun()
                                    

                        # # FINAL_RAG_QUERY append_query_with_acronym
                        if concatenated_definitions.strip() != "":
                            concatenated_definitions_summary = st.session_state.llm(f'''From the <TEXT>, extract all the given ACRONYMS and their definitions. 
                                                                                    Donot add or include other details. If no ACRONYMS, return ''. Here is the <TEXT> : 
                                                                                    {concatenated_definitions}
                                                                                    ''')
                            FINAL_RAG_QUERY = FINAL_RAG_QUERY + " . " +  concatenated_definitions_summary
                            
                        # Check and Append today' datetime if needed
                        my_template = configs['prompts']['DATETIME_REQUIREMENT_CHECK_template']['template']
                        my_prompt = PromptTemplate(input_variables=["user_query"],template=my_template)
                        my_formatted_prompt = my_prompt.format(user_query=FINAL_RAG_QUERY)
                        my_out = st.session_state.llm_query_improve(my_formatted_prompt)
                        DATETIME_FLAG = my_out
                        if contains_word(DATETIME_FLAG,"True"):   
                            current_datetime = date.today()
                            FINAL_RAG_QUERY = FINAL_RAG_QUERY + f". Today is {current_datetime}"                


                        # Replace Single Quotes in String with empty string
                        FINAL_RAG_QUERY = FINAL_RAG_QUERY.replace("'"," ")                    

                        print_wide_line_message(
                            f"FINAL_RAG_QUERY:\n {FINAL_RAG_QUERY}")
                        
                        # Process query, use only double quotes "'xxx'"->"xxx" or '"xxx"'->"xxx"
                        FINAL_RAG_QUERY = normalize_quotes(FINAL_RAG_QUERY)  

                        user_input = FINAL_RAG_QUERY     
                                        
                        
                        #breakpoint()

                        with st.spinner(f'''Final Search Query: {FINAL_RAG_QUERY}'''):
                            # Append final query to chat history and display
                            # st.session_state.chat_history.append({"role": "assistant", "message": FINAL_RAG_QUERY})
        
                            # ToFix: No longer used.
                            if configs['generate']['enable_dynamic_vec_db_selection'] == True:
                                if st.session_state.faiss_persist_directory == "./faiss/ALL_V2":
                                    # USE LLM to find the best vectorstore based on user_input
                                    my_template = configs["prompts"]["llm_identify_db_template"]["template"]
                                    my_prompt = PromptTemplate(input_variables=["user_input"], template=my_template)
                                    llm_identify_db_template = my_prompt.format(user_input=user_input)
                                    # EXECUTE CODE
                                    VecDB_TYPE = st.session_state.llm(llm_identify_db_template)
                                    print(VecDB_TYPE)
                                    VecDB_TYPE = str(VecDB_TYPE).strip()
                                    VecDB_TYPE = VecDB_TYPE.strip("\n")
                                    # use fuzzy closest match
                                    VecDB_TYPE_ACTUAL = process.extractOne(
                                        VecDB_TYPE, configs['data']['vec_store_options'])
                                    # ToFIX:CODE-NOT-ELEGANT
                                    VecDB_TYPE_ACTUAL = VecDB_TYPE_ACTUAL[0]
                                else:
                                    VecDB_TYPE_ACTUAL = process.extractOne(
                                        st.session_state.faiss_persist_directory, configs['data']['vec_store_options'])
                                    VecDB_TYPE_ACTUAL = VecDB_TYPE_ACTUAL[0]
                            else:
                                VecDB_TYPE_ACTUAL = process.extractOne(
                                    st.session_state.faiss_persist_directory, configs['data']['vec_store_options'])
                                VecDB_TYPE_ACTUAL = VecDB_TYPE_ACTUAL[0]
        
                            print(f"<BEGIN>SELECTION:{VecDB_TYPE_ACTUAL}<END>")
                            print(f"Most appropriate Vector database:{VecDB_TYPE_ACTUAL}")
                            # get appropraite retrievers & qa_chain
                        
                            router_out_GraceDB_FLAG = "False"
                          
                                

                            if True:
                                _check_stop_search()

                                response_result_returned_FLAG = False
                                  
                            
                                if response_result_returned_FLAG == False:
                                    # check if only System-1 thinking is required
                                    print_wide_line_message(
                                        "Checking if System-1 Response is sufficient")
                                    with traced_spinner("Activating System-1 thinking", detail="Attempting a fast LLM-only answer (no retrieval)."):

                                        if st.session_state.search_type == "Standard":
                                            CONSISTENT_FLAG, llm_out_avg = fetch_system1_OUTPUT(
                                                user_input)

                                            
                                            if CONSISTENT_FLAG == False:
                                                print(
                                                    "\nCONSISTENT_FLAG returned False on first try. Trying once more.\n")
                                                CONSISTENT_FLAG, llm_out_avg = fetch_system1_OUTPUT(
                                                    user_input)
                                            
                                                if CONSISTENT_FLAG == False:
                                                    llm_out_avg = "please try again" # just a place holder text
                                        else:
                                            CONSISTENT_FLAG = False

                                    # Trace the System-1 vs System-2 decision
                                    try:
                                        if st.session_state.search_type != "Standard":
                                            trace_add("System-1 skipped", detail=f"search_type={st.session_state.search_type}")
                                        elif CONSISTENT_FLAG == True and len(str(llm_out_avg).strip()) != 0:
                                            trace_add("System-1 sufficient", detail="Using fast answer without retrieval.")
                                        else:
                                            trace_add("System-1 insufficient", detail="Falling back to System-2 retrieval (RAG).")
                                    except Exception:
                                        pass

                                    if CONSISTENT_FLAG == True and len(str(llm_out_avg).strip())!=0:
                                        print_wide_line_message(
                                            "System-1 Response is sufficient")
                                        response_result = llm_out_avg + "\n\n" + f'''**Note:** This is a quick response generated by a language model, which may contain inaccuracies. 
                                                                        For more detailed answers, please rephrase your question and be sure to request sources and supporting documents.
                                                                        '''
                                                        
                                        print_wide_line_message("System-1 Output")
                                        print_wide_line_message(llm_out_avg)                                
                                    else:
                                        _check_stop_search()
                                        print_wide_line_message(
                                            "Switching to System-2 Response")
                                        

                                        # Fall-back to Full-RAG (System-2)
                                        print(
                                            'Executing: response = st.session_state.qa_chain(user_input)  \n')
                                        print('st.session_state.prompt\n')
                                        print(st.session_state.prompt)



                                        if configs['generate']['enable_superposition_answer'] == True:
                                            with traced_spinner("Switched to System-2 thinking", detail="Running retrieval + synthesis (RAG)."):
                                                
                                        
                                                trace_add(
                                                    "RAG search started",
                                                    detail=f"search_type={st.session_state.search_type} • query={FINAL_RAG_QUERY[:200]}"
                                                )
                                                OUT = RAG_STAR(FINAL_RAG_QUERY,ensembele_superposition_answer,search_type=st.session_state.search_type )
                                                #response_result, QA_ANSWER, RAPTOR_SUMMARY, MORE_DETAILS,filtered_retrieved_docs_LLM_answers,QA_ANSWER_AGGREGATOR = ensembele_superposition_answer(user_input)
                                                response_result = OUT['final_answer']   
                                                

                                                if st.session_state.search_type == "Standard":
                                                    if len(response_result)  < 3 or response_result == "\n\n":
                                                        if configs['retrieval']['ENABLE_AUTO_SWITCH_TO_DEEPSEARCH']:
                                                            trace_add("Auto-switch to DeepSearch", detail="Standard search returned no/empty answer; escalating search depth.")
                                                            OUT = RAG_STAR(FINAL_RAG_QUERY,ensembele_superposition_answer,search_type="DeepSearch",skip_baseline_answer=True )
                                                            response_result = OUT['final_answer']
                                                        else:
                                                            trace_add("No relevant info found", detail="Standard search empty and auto-switch disabled; requesting clarification/rephrase.")
                                                            # Tell user to modify query or use DeepSearch
                                                            response_result = "Oops! I am unable to find any relevant information. Could you please try again by rephrasing the question with more context. Else enable DeepSearch and try again."

                                                    
                                                                                        

                                        else:
                                            response = st.session_state.qa_chain(
                                                user_input)
                                            print("\nqa_chain response\n")
                                            print(response)
                                            response_result = response['result']

                                        ##################
  
                                        if configs['retrieval']['enable_verbose']:
                                            diagnostic_print_RAG_chunks(
                                                st, user_input, st.session_state.max_return_docs)
        

                            # DISPLAY OUTPUT
                            # streamlit markdown write output
                            if router_out_GraceDB_FLAG == False and CONSISTENT_FLAG == False:
                                pass
                            else:
                                # Display the assistant's response and persist it
                                trace_add("Final answer delivered", detail=f"Length: {len(str(response_result))} characters")
                                streamlit_add_msg(
                                    st=st,
                                    role="assistant",
                                    message=response_result,
                                    persist=True
                                )
                                # Attach and render reasoning trace for this assistant turn
                                try:
                                    _trace_items = copy.deepcopy(st.session_state.get("reasoning_trace", []))
                                    attach_trace_to_last_assistant_message(_trace_items)
                                    render_reasoning_trace_expander(_trace_items)
                                except Exception:
                                    pass

                                # Persist the assistant's final reply
                                if st.session_state.get("current_conv_id") is not None:
                                    save_message(st.session_state.current_conv_id, "assistant", response_result)
                                    if st.session_state["is_guest"] == True and st.session_state["guest_private"] == False:
                                        auto_generate_conversation_title()
                                        # Stay in the same conversation on rerun
                                        
                                st.session_state.refresh_selected_conv = True
                                maybe_save_turn_to_memory(user_input, response_result)
                                st.session_state["_pp_prev_answer"] = response_result  # added

                                if "_pp_prev_significant_answer" not in st.session_state:
                                    st.session_state["_pp_prev_significant_answer"] = response_result
                                if len(response_result) > 3000:
                                    st.session_state["_pp_prev_significant_answer"] = response_result  # added

                else:
                    full_response = ""
                    message_placeholder.markdown(full_response)
                    response = {}  # Initialize an empty dictionary
                    # Set the 'result' key to an empty string
                    response['result'] = ""

        # early stop exit
        except SearchCancelled:
            cancel_msg = "⛔ Search cancelled."

            # show a normal assistant message instead of a traceback
            message_placeholder.markdown(cancel_msg)

            # persist it the same way you persist other assistant messages
            st.session_state.chat_history.append({"role": "assistant", "message": cancel_msg})
            if st.session_state.get("current_conv_id") is not None:
                save_message(st.session_state.current_conv_id, "assistant", cancel_msg)

            # optional: add to your reasoning trace, if you keep one
            try:
                trace_add("Stop pressed", "SearchCancelled raised; aborting run")
            except Exception:
                pass

            st.stop()                    