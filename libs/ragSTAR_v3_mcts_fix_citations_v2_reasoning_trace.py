"""
RAG-Star with Smart Caching & Fixed MCTS  (v5-full)

Drop-in for libs/ragSTAR_v3.py
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from itertools import cycle
import traceback

import asyncio, inspect
import math
import pickle
import random
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
import time, threading, re
from contextlib import contextmanager
import groq 
import httpx




import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from libs.print import streamlit_add_msg,pretty_print_docs

from config import config
import libs.summarizer_v3 # ensure module is in sys.modules before get_type_hints
from libs.summarizer_v3 import langgraph_summarize_documents,langgraph_task_specific_document_processing
from libs.regex import extract_answer_or_raw


# -----------------------------------------------------------------------------
# Reasoning trace hooks (high-level). Safe to display to users.
# -----------------------------------------------------------------------------
def _trace_add(title: str, detail: str = "") -> None:
    try:
        if "reasoning_trace" not in st.session_state or not isinstance(st.session_state.reasoning_trace, list):
            st.session_state.reasoning_trace = []
        max_items = int(st.session_state.get("reasoning_trace_max_items", 120))
        if len(st.session_state.reasoning_trace) >= max_items:
            return
        st.session_state.reasoning_trace.append({
            "title": str(title).strip(),
            "detail": str(detail).strip() if detail else ""
        })
    except Exception:
        pass


@contextmanager
def _traced_spinner(label: str, detail: str = ""):
    _trace_add(label, detail)
    with st.spinner(label):
        yield




# ❶ a tiny round-robin iterator over the two LLM objects
llms = cycle([  st.session_state.llm_2,
                st.session_state.llm_3,
                st.session_state.llm_4,
                st.session_state.llm_5])
    

# ╭────────────────────────── configuration ─────────────────────────╮
CFG = config.load_config()


# MCTS parameters
MCTS_CFG = CFG["retrieval"]
MAX_ITER = MCTS_CFG["mcts_rag_max_iterations"]
MAX_CHILD = MCTS_CFG["max_children_per_node"]
EXPL_C = MCTS_CFG["mcts_rag_exploration_constant"]
EARLY_RDS = min(MCTS_CFG["mcts_early_stopping_rounds"], MAX_ITER)
EARLY_THR = MCTS_CFG["mcts_early_stopping_threshold"]

# cache parameters
CACHE_PATH = Path(MCTS_CFG["DEEPSEARCH_CACHE_PATH"])
CACHE_SIM = float(MCTS_CFG["DEEPSEARCH_CACHE_SIMILARITY_THRESH"])

# embedder
_ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
# ╰──────────────────────────────────────────────────────────────────╯


# ╭────────────────────────── helpers ───────────────────────────────╮
def _embed(text: str) -> np.ndarray:
    """Return L2-normalised embedding vector."""
    return _ST_MODEL.encode([text], normalize_embeddings=True)[0]


def _standardise_result(res: Any) -> Dict[str, Any]:
    """
    Accept either the 6-tuple or the already-dict return shape produced by
    your ensemble function and convert to a canonical dict.
    """
    if isinstance(res, dict):
        return res
    if isinstance(res, tuple) and len(res) == 6:
        (
            response_result,
            QA_ANSWER,
            WIKI_SUMMARY,
            MORE_DETAILS,
            filt_docs,
            QA_AGG,
        ) = res
        return {
            "response_result": response_result,
            "QA_ANSWER": QA_ANSWER,
            "WIKI_SUMMARY": WIKI_SUMMARY,
            "MORE_DETAILS": MORE_DETAILS,
            "filtered_retrieved_docs_LLM_answers": filt_docs,
            "QA_ANSWER_AGGREGATOR": QA_AGG,
        }
    raise TypeError("Unexpected result format from superposition_fn")





# ───────────────────────── cache helpers ──────────────────────────
def _load_cache() -> List[Tuple[np.ndarray, str, Dict[str, Any]]]:
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as fh:
            raw = pickle.load(fh)
            # basic shape safety check
            return [e for e in raw if isinstance(e, tuple) and len(e) == 3]
    return []


def _save_cache(cache) -> None:
    with open(CACHE_PATH, "wb") as fh:
        pickle.dump(cache, fh)


def cached_superposition(
    super_fn: Callable[[str], Any], sim_thresh: float = CACHE_SIM
) -> Callable[[str], Dict[str, Any]]:
    """
    Decorator: wrap slow RAG ensemble with semantic-similarity cache.
    """

    #if getattr(super_fn, "_is_cached", False):  # already wrapped
    #    return super_fn

    cache = _load_cache()
    _trace_add("DeepSearch cache loaded", detail=f"entries={len(cache)} • path={CACHE_PATH}")
    if CFG['retrieval']['DEEPSEARCH_STREAMLIT_CACHE_VERBOSE']:
        st.write(f"[cache] Loaded {len(cache)} entries from {CACHE_PATH}")

    def _wrapper(query: str) -> Dict[str, Any]:
        q_vec = _embed(query)

        best_sim, best_res = 0.0, None
        for vec, _, res in cache:
            sim = float(np.dot(vec, q_vec))
            if sim > best_sim:
                best_sim, best_res = sim, res


        if best_sim >= sim_thresh and best_res['filtered_retrieved_docs_LLM_answers'] != []:
            _trace_add("Cache HIT", detail=f"sim={best_sim:.3f} • bypassed retriever")
            if CFG['retrieval']['DEEPSEARCH_STREAMLIT_CACHE_VERBOSE']:
                st.write(f"[cache] ✅ HIT (sim={best_sim:.3f}) – bypassing retriever")
            _trace_add("Cache HIT", detail=f"sim={best_sim:.3f} • bypassed retriever")
            return _standardise_result(best_res)


        # cache miss
        _trace_add("Cache MISS", detail="running ensemble retriever")
        if CFG['retrieval']['DEEPSEARCH_STREAMLIT_CACHE_VERBOSE']:
            st.write("[cache] ❌ MISS – running ensemble retriever")
        res = _standardise_result(super_fn(query))

        cache.append((q_vec, query, res))
        _save_cache(cache)
        _trace_add("Cache saved", detail=f"size={len(cache)}")
        if CFG['retrieval']['DEEPSEARCH_STREAMLIT_CACHE_VERBOSE']:
            st.write(f"[cache] ➕ saved (size={len(cache)})")
        return res


    _wrapper._is_cached = True  # type: ignore
   
    return _wrapper




# ─────────────────── citations helpers (generic URLs) ───────────────────
import urllib.parse


# ─────────────────── LLM-based planner-aware subquery generator ───────────────────
import json as _json_mod
import re as _re_mod
from typing import List as _List, Optional as _Optional

def _rag_norm(_s: str) -> str:
    return _re_mod.sub(r'\s+', ' ', (_s or '').strip()).lower()

def _rag_parse_json_list(_s: str) -> _List[str]:
    try:
        obj = _json_mod.loads(_s)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    # Fallback: parse lines
    out = []
    for ln in str(_s).splitlines():
        ln = _re_mod.sub(r'^\s*(?:[-*•\d\.\)\(]+)\s*', '', ln).strip()
        if ln:
            out.append(ln)
    return out

def llm_planner_subqueries(
    root_query: str,
    parent_query: str,
    context_text: str,
    existing: _List[str],
    *,
    llm_fn: _Optional[callable] = None,
    max_out: int = 6,
) -> _List[str]:
    """
    LLM-driven planner:
      • Detects entities/topics; proposes DISTINCT, FACET-DIVERSE sub-questions.
      • Avoids duplicates using existing siblings and global history injected into context_text.
      • Does NOT hardcode journals or topics; infers from inputs.
    Output: list[str]
    """
    prev_asked = []
    m = _re_mod.search(r'__PREV_ASKED__\s*(.*)$', context_text, flags=_re_mod.S)
    if m:
        prev_asked = [_re_mod.sub(r'^\s*[-*•]\s*', '', ln).strip()
                      for ln in m.group(1).splitlines() if ln.strip()]
    existing_norm = {_rag_norm(q) for q in (existing or [])}
    history_norm  = {_rag_norm(q) for q in (prev_asked or [])}

    if llm_fn is None:
        try:
            import streamlit as _st_mod
            llm_fn = _st_mod.session_state.llm
        except Exception:
            raise RuntimeError("llm_planner_subqueries requires `llm_fn` or st.session_state.llm")

    ctx = context_text[:2500]
    prev = "\n".join(f"- {q}" for q in prev_asked[:40])

    prompt = f"""
You are a planning assistant for multi-hop scientific question answering.
Propose DISTINCT, TARGETED sub-questions that, if answered, would materially help to answer the user's
original query. Do not assume domain specifics (journals, subfields); infer from inputs.

REQUIREMENTS
1) DECOMPOSE if multiple named entities (people, projects, instruments, datasets, facilities, collaborations,
   topics) are present. Ensure coverage ACROSS entities, not only joint phrasing.
2) DIVERSIFY FACETS: contributions/findings; methods/instrumentation; datasets/evidence; collaborators;
   chronology (early vs recent); impact/validation/metrics; comparisons/contrasts; and one integrative
   “synthesis” question if budget allows.
3) AVOID DUPLICATES (semantic paraphrases): Do NOT repeat anything listed under `Existing siblings` or
   `Previously asked in the tree`.
4) OUTPUT FORMAT: return ONLY a valid JSON array of strings. No prose.

Inputs
------
Original query:
{root_query}

Parent sub-question:
{parent_query}

Context snippets (may include notes/citations):
{ctx}

Existing siblings (avoid):
{chr(10).join("- " + q for q in existing)}

Previously asked in the tree (avoid):
{prev}

Budget (max items): {max_out}

Output
------
A pure JSON array of strings.
""".strip()

    raw = llm_fn(prompt) or ""
    cands = _rag_parse_json_list(str(raw))

    out, seen = [], set()
    for q in cands:
        n = _rag_norm(q)
        if not n: 
            continue
        if n in seen or n in existing_norm or n in history_norm:
            continue
        seen.add(n)
        out.append(q.strip())
        if len(out) >= max_out:
            break

    if not out:
        fallback = [
            f"Break down the question into entities/topics and propose targeted lookups relevant to: {root_query}",
            f"List the most influential works and supporting evidence relevant to: {root_query}",
            f"Identify collaborators, methods/instruments, datasets, and validation relevant to: {root_query}",
        ]
        for q in fallback:
            n = _rag_norm(q)
            if n in existing_norm or n in history_norm:
                continue
            out.append(q)
            if len(out) >= max_out:
                break
    return out[:max_out]


def _normalize_url_for_cite(u: str) -> str:
    if not u: return ''
    u = u.strip()
    if not re.match(r'^[a-z]+://', u, flags=re.I):
        u = 'https://' + u
    p = urllib.parse.urlparse(u)
    scheme, netloc, path = p.scheme.lower(), p.netloc.lower(), p.path or ''
    if netloc.endswith(':80') and scheme == 'http': netloc = netloc[:-3]
    if netloc.endswith(':443') and scheme == 'https': netloc = netloc[:-4]
    path = re.sub(r'/+', '/', path)
    if len(path) > 1 and path.endswith('/'): path = path[:-1]
    # strip tracking params
    q = [(k,v) for (k,v) in urllib.parse.parse_qsl(p.query, keep_blank_values=True) if not k.lower().startswith('utm_')]
    query = urllib.parse.urlencode(sorted(q), doseq=True)
    return urllib.parse.urlunparse((scheme, netloc, path, '', query, ''))

_URL_RX = re.compile(r'https?://[^\s)>\]]+', re.I)

def _extract_urls_from_text(txt: str) -> list[str]:
    if not isinstance(txt, str) or not txt: return []
    urls = []
    for u in _URL_RX.findall(txt):
        try:
            urls.append(_normalize_url_for_cite(u))
        except Exception:
            continue
    # dedupe preserving order
    seen = set(); out = []
    for u in urls:
        if u in seen: continue
        seen.add(u); out.append(u)
    return out

def _inline_marker(keys: list[str]) -> str:
    if not keys: return ''
    # use stable string keys (url:...)
    quoted = ['"url:' + k + '"' for k in keys]
    return ' [@' + ",".join(quoted) + ']'

# ──────────────────── small evaluator prompts ─────────────────────
def _eval_answer(ans: str, subq: str) -> float:
    prompt = (
        f"Rate the quality of the answer (1-3). "
        f"Q: {subq}\nA: {ans}\nReturn <score>n</score>"
    )
    resp = st.session_state.llm(prompt)
    m = re.search(r"<score>(\d(?:\.\d+)?)</score>", resp)
    return float(m.group(1)) / 3 if m else 0.33


def _eval_query(subq: str, ctx: str) -> float:
    prompt = (
        f"Is this follow-up query logically consistent with context? "
        f"Return 0 or 1 in <score> tag.\nContext:\n{ctx}\nQuery:{subq}"
    )
    resp = st.session_state.llm(prompt)
    m = re.search(r"<score>(\d(?:\.\d+)?)</score>", resp)
    return float(m.group(1)) if m else 0.0


#def _gen_subquery(ctx: str, existing: List[str]) -> str:
#    extra = f"Existing: {', '.join(existing)}\n" if existing else ""
#    prompt = (
#        f"{ctx}\n{extra}"
#        "Generate ONE distinct follow-up query to refine the answer. "
#        "Give *only* the query."
#    )
#    return st.session_state.llm(prompt).strip()
#

def _gen_subquery(existing_context: str, existing_subqueries: List[str],root_query) -> str:    
    #context = f"Existing Context: {', '.join(existing_context)}\n" if existing_queries else ""
    #previous_queries = f"Existing Queries: {', '.join(existing_queries)}\n" if existing_queries else ""
    #prompt = (
    #    f"{existing_context}\n{previous_queries}"
    #    "Generate ONE distinct follow-up query to refine the answer. "
    #    "Give *only* the query."
    #)
    template = CFG['prompts']['DEEPSEARCH_SUBQUERY_template']['template']
    prompt = PromptTemplate(
        input_variables=["root_query", "existing_context","existing_subqueries"],
        template=template
    )
    formatted_prompt= prompt.format(root_query=root_query,existing_context=existing_context,existing_subqueries=existing_subqueries)
    llm_out = st.session_state.llm(formatted_prompt).strip()
    distinct_subquery = extract_answer_or_raw(llm_out, pattern="DISTINCT_SUBQUERY")
    return distinct_subquery

def _get_context(node: "RAGStarNode") -> str:
    out, cur = [], node
    while cur:
        out.append(f"Subquery: {cur.subquery}\nAnswer: {cur.answer or ''}")
        cur = cur.parent
    return "\n".join(reversed(out))



@contextmanager
def use_llm_temporarily(temp_llm):
    """
    Replace st.session_state.llm with `temp_llm` just for the `with` block.
    Restores the original object afterwards.
    """
    import streamlit as st
    original = st.session_state.llm
    st.session_state.llm = temp_llm
    try:
        yield
    finally:
        st.session_state.llm = original

# ╭──────────────────────────── RAG-Star Node ──────────────────────╮
class RAGStarNode:
    def __init__(
        self, subquery: str, answer: str | None = None, parent: "RAGStarNode" | None = None
    ):
        self.subquery = subquery
        self.answer = answer
        self.parent = parent
        self.children: List["RAGStarNode"] = []

        self.visits = 0
        self.value = 0.0
        self.score = 0.0
        self.details: Dict[str, Any] = {}
        self.is_expanded = False

        self._emb = _embed(subquery)

    @property
    def query(self) -> str:
        """Return the (sub-)query held by this node."""
        return self.subquery
    @property
    def root_query(self) -> str:
        """Return the original root query."""
        node = self
        while node.parent is not None:
            node = node.parent
        return node.subquery        

    # --- UCT helpers -------------------------------------------------
    def average_value(self) -> float:
        return self.value / self.visits if self.visits else 0.0

    def best_child(self) -> "RAGStarNode":
        return max(
            self.children,
            key=lambda ch: ch.average_value()
            + EXPL_C * math.sqrt(math.log(self.visits + 1) / (ch.visits + 1e-9)),
        )

    # --- tree ops ----------------------------------------------------
    def expand(
        self,
        n_children: int,
        super_fn: Callable[[str], Dict[str, Any]],
        global_embeddings: List[np.ndarray],
        *,
        dup_thresh: float = CFG['retrieval']['REGEX_DEDUP_PARAGRAPH_SIMILARITY_THRESHOLD'],
    ):
        """Generate up to n_children distinct follow-up nodes."""
        if self.is_expanded:
            return

        ctx = _get_context(self)
        existing_prompts: List[str] = []
        attempts = 0
        while len(self.children) < n_children and attempts < n_children * 4:
            attempts += 1
            subq = _gen_subquery(ctx, existing_prompts,self.root_query)
            q_vec = _embed(subq)
            # duplicate suppression across entire search
            if any(float(np.dot(q_vec, v)) >= dup_thresh for v in global_embeddings):
                continue

            global_embeddings.append(q_vec)
            existing_prompts.append(subq)

            ans_det = _standardise_result(super_fn(subq))
            ans_txt = f"{ans_det['QA_ANSWER']}\n\n{ans_det['WIKI_SUMMARY']}"
            reward = _eval_answer(ans_txt, subq) * _eval_query(subq, ctx)

            with st.expander(f"🔍 Response to follow-up query {len(global_embeddings)}: `{subq}`"):
                st.markdown(ans_txt)

            with st.expander("Retrieved Documents"):
                #streamlit_add_msg(st,role="assistant-0",message=f"current_query:{subq}") 
                #streamlit_add_bold_heading(st,role="assistant-0",message="Retrieved Documents")   
                streamlit_add_msg(st,role="assistant-0",message=ans_det['MORE_DETAILS'])    



            child = RAGStarNode(subquery=subq, answer=ans_txt, parent=self)
            child.score = reward
            child.details = ans_det
            self.children.append(child)

        self.is_expanded = True

    def backpropagate(self, reward: float) -> None:
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(reward)


# ╭──────────────────────── path utilities ─────────────────────────╮
def _collect_paths(root: RAGStarNode) -> List[Dict[str, Any]]:
    paths: List[Dict[str, Any]] = []

    def dfs(node: RAGStarNode, stack: List[RAGStarNode], score: float):
        stack.append(node)
        score += node.score
        if not node.children:
            paths.append(
                {
                    "path_subqueries": [n.subquery for n in stack if n.parent],
                    "path_answers": [n.answer for n in stack if n.parent],
                    "cumulative_score": score,
                    "path_nodes": [n for n in stack if n.parent],
                }
            )
        else:
            for ch in node.children:
                dfs(ch, stack, score)
        stack.pop()

    dfs(root, [], 0.0)
    return paths



def _compose_final_answer(path_answers: List[str], query: str, baseline: str) -> str:
    joined = "\n\n".join(path_answers)

    template = CFG['prompts']['UPDATE_REPORT_WITH_NEW_INFO_template']['template']
    prompt = PromptTemplate(
        input_variables=["user_query", "existing_answer","additional_context"],
        template=template
    )
    formatted_prompt= prompt.format(user_query=query,existing_answer=baseline,additional_context=joined)
    return st.session_state.llm(formatted_prompt).strip()

def run_sync(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)           # ← no loop: safe path
    if loop.is_running():
        # ✔ create a *new* private loop instead of re-using Streamlit’s
        new_loop = asyncio.new_event_loop()
        return new_loop.run_until_complete(coro)
    return loop.run_until_complete(coro)


def run_with_model_swap(coro_builder, *, max_tries=3):
    global llms
    last_exc = None

    for attempt in range(1, max_tries + 1):
        llm_current = next(llms)
        print(f"    attempt {attempt}/{max_tries} using {llm_current.model_name}")

        try:
            # ⬇️ temporarily make this the default LLM for *all* downstream calls
            with use_llm_temporarily(llm_current):
                return run_sync(coro_builder(llm_current))
        except (asyncio.TimeoutError, ConnectionError,
                TimeoutError, groq.APITimeoutError, httpx.PoolTimeout) as e:
            print("    ⚠️  timeout:", e)
            last_exc = e
            continue
        except Exception:
            print("    ❌ unexpected error:\n", traceback.format_exc())
            raise

    raise last_exc or RuntimeError("All model attempts failed")



# ╭──────────────────────── main runner ────────────────────────────╮
def run_RAGSTAR(
    query: str,
    superposition_fn: Callable[[str], Any],
    *,
    search_type: str = "DeepSearch",
    skip_baseline_answer: bool = False,
    **kwargs
) -> Dict[str, Any]:

    # Always get a fresh wrapper depending on flag
    try:
        if CFG['retrieval']['ENABLE_CACHE'] and st.session_state.cache_enabled:
            super_fn = cached_superposition(superposition_fn)
        else:
            super_fn = superposition_fn
    except:
        super_fn = cached_superposition(superposition_fn)

        _trace_add(
            "RAG_STAR invoked",
            detail=f"query={query[:200]} • search_type={search_type} • skip_baseline={skip_baseline_answer}"
        )



    # baseline
    if skip_baseline_answer:
        baseline_answer = ""
        baseline_details = {}
    else:
        baseline_details = _standardise_result(super_fn(query))
        baseline_answer = (
            f"{baseline_details['QA_ANSWER']}\n\n{baseline_details['WIKI_SUMMARY']}"
        )

        
 
    # Record baseline stats in reasoning trace
    try:
        _verified = len((baseline_details or {}).get("filtered_retrieved_docs_LLM_answers") or [])
        _trace_add("Baseline search complete", detail=f"verified_passages={_verified}")
    except Exception:
        pass

    # Streamlit UI
    st.info("Answer from Standard Search")
    with st.expander("🔍 Baseline Answer: Click to view"):
        st.markdown(baseline_answer or "_(skipped)_")

    if baseline_details != {}:
        with st.expander("Retrieved Documents"):
            streamlit_add_msg(st,role="assistant-0",message=f"current_query:{query}") 
            #streamlit_add_bold_heading(st,role="assistant-0",message="Retrieved Documents")   
            streamlit_add_msg(st,role="assistant-0",message=baseline_details['MORE_DETAILS'])            

    if search_type not in ("DeepSearch", "UltraDeepSearch"):
        _trace_add("Returning baseline answer", detail=f"search_type={search_type}")
        return {
            "final_answer": baseline_answer,
            **baseline_details,
        }

    # adjust depth parameters 

    iter_limit =  MAX_ITER
    child_limit = MAX_CHILD
 

    # root
    root = RAGStarNode(subquery=query, answer=baseline_answer, parent=None)
    root.details = baseline_details
    global_embs = [_embed(query)]

    st.info("🔄 Generating DeepSearch follow-ups …")
    root.expand(child_limit, super_fn, global_embs)

    # Trace: list the initial follow-up queries
    try:
        _followups = [getattr(c, "subquery", "") for c in (root.children or [])]
        _followups = [q for q in _followups if isinstance(q, str) and q.strip()]
        if _followups:
            _trace_add(
                "DeepSearch follow-ups generated",
                detail="\n".join([f"- {q}" for q in _followups[:max(1, child_limit)]])
            )
    except Exception:
        pass

    # MCTS loop
    cumulative_best_scores: List[float] = []
    no_improve_rounds = 0

    _trace_add("MCTS search started", detail=f"iter_limit={iter_limit} • child_limit={child_limit} • early_rounds={EARLY_RDS} • early_thr={EARLY_THR}")

    for it in range(iter_limit):
        st.info(f"### Search iteration {it + 1}/{iter_limit}")

        # Selection
        node = root
        trace: List[RAGStarNode] = []
        while node.is_expanded and node.children:
            trace.append(node)
            node = node.best_child()

        # Expansion (if not already expanded)
        if not node.is_expanded:
            node.expand(child_limit, super_fn, global_embs)

        # Pick one fresh child for simulation
        # ── inside run_RAGSTAR, just before the line that errored ─────────
        fresh = [c for c in node.children if c.visits == 0]
        # If expand() produced no children at all, bail out gracefully
        if not node.children:
            st.warning("⚠️  Expansion produced no children — stopping search early.")
            _trace_add("MCTS stopped early", detail="Expansion produced no children.")
            break            # or `continue`, depending on whether you want the loop to end
        # If all children were already visited, pick any (old) child
        if not fresh:
            fresh = node.children
        sim_node = random.choice(fresh)
        reward = sim_node.score

        # Back-prop
        sim_node.backpropagate(reward)

        # Early stopping bookkeeping
        paths = _collect_paths(root)
        best_score = max(p["cumulative_score"] for p in paths) if paths else 0
        cumulative_best_scores.append(best_score)
        if len(cumulative_best_scores) > 1:
            if (cumulative_best_scores[-1] - cumulative_best_scores[-2]) < EARLY_THR:
                no_improve_rounds += 1
            else:
                no_improve_rounds = 0
            if no_improve_rounds >= EARLY_RDS:
                st.warning("⏹ Early stopping triggered.")
                _trace_add("Early stopping triggered", detail=f"iteration={it+1} • best_score={best_score:.3f}")
                break

    # final aggregation
    st.info(f"### DeepSearch Answer")

    with _traced_spinner("Synthesing DeepSearch Answer", detail="Aggregating explored paths and composing final answer."):
        all_paths = _collect_paths(root)
        _trace_add("DeepSearch paths collected", detail=f"paths={len(all_paths)}")
        all_paths_sorted = sorted(all_paths, key=lambda d: -d["cumulative_score"])

        # compose answers for each path
        for p in all_paths_sorted:
            p["final_answer"] = _compose_final_answer(
                p["path_answers"], query, baseline_answer
            )

        # choose best path’s final answer
        best_final_answer = all_paths_sorted[0]["final_answer"]
        try:
            _best_score = float(all_paths_sorted[0].get("cumulative_score", 0.0)) if all_paths_sorted else 0.0
            _trace_add("DeepSearch best path selected", detail=f"best_score={_best_score:.3f}")
        except Exception:
            pass

        #  LangGraph summarisation  
        _trace_add("Final report method", detail=str(CFG["retrieval"].get("DEEPSEARCH_FINAL_REPORT_METHOD")))
        if CFG["retrieval"].get("DEEPSEARCH_FINAL_REPORT_METHOD") == "langgraph_based":
            # --- hot-reload safety net -----------------------------------------
            import importlib, sys
            if "libs.summarizer" not in sys.modules:
                sys.modules["libs.summarizer"] = importlib.import_module("libs.summarizer")
            # -------------------------------------------------------------------


            # Get Docs
            
            # Collect citations per path by scanning MORE_DETAILS of nodes in each path
            citations_per_path = []
            global_citations = []
            for p in all_paths_sorted:
                keys = []
                nodes = p.get("path_nodes") or []
                for n in nodes:
                    md = (getattr(n, "details", {}) or {}).get("MORE_DETAILS", "")
                    keys.extend(_extract_urls_from_text(md))
                # dedupe
                seen = set(); uniq = []
                for k in keys:
                    if k in seen: continue
                    seen.add(k); uniq.append(k)
                citations_per_path.append(uniq)
                for k in uniq:
                    if k not in global_citations:
                        global_citations.append(k)

            # Build documents with inline markers to help the LLM retain citations
            docs = []
            for p, keys in zip(all_paths_sorted, citations_per_path):
                marker = _inline_marker(keys)
                docs.append(Document(page_content=(p["final_answer"] + ("\n\n" + marker if marker else "")),
                                    metadata={"citations": keys}))


            print("Docs used for LangGraph summarisation")
            pretty_print_docs(docs)
        

            parts_out = {}
            enable_groq_TFM_check = st.session_state.GROQ_TOKENS_TFM_CHECK

            # order you want the sections to appear
            display_order = [
                "KEY_FACTS",
                "FINER_DETAILS",
                "SUMMARY",
                "REFERENCES",
    ]
            # Original code without model switching
            #for part in display_order:
            #    llm_current = next(llms)                       # ❷ pick one LLM for this turn
            #    print(f"\n▶ {part}  (using {llm_current.__class__.__name__})")
            #
            #    with st.spinner(f"Preparing DeepSearch {part}"):
            #        result = run_sync(
            #            langgraph_task_specific_document_processing(
            #                llm_current,           # ❸ pass it in
            #                docs,
            #                query,
            #                task_type=part,
            #                enable_groq_TFM_check = enable_groq_TFM_check
            #            )
            #        )
            #        parts_out[part] = result 

            # Deterministic REFERENCES constructed from collected citations
            if global_citations:
                ref_lines = []
                for i, u in enumerate(global_citations, 1):
                    ref_lines.append(f"[{i}] {u}")
                parts_out["REFERENCES"] = "\n".join(ref_lines)
            else:
                parts_out["REFERENCES"] = "_No references._"
            #        print(f"  ↳ {len(result.split())} words")                        
            #        with st.expander(f"🔍 Click to view DeepSearch {part}"):            
            #            st.markdown(result)

            for part in display_order:
                print(f"\n▶ {part}")
                with st.spinner(f"Preparing DeepSearch {part}"):
                    try:
                        result = run_with_model_swap(
                            lambda llm_cur: langgraph_task_specific_document_processing(
                                llm_cur, docs, query,
                                task_type=part,
                                enable_groq_TFM_check=enable_groq_TFM_check,
                            ),
                            max_tries=3,
                        )
                    except Exception as e:
                        result = f"**ERROR for {part}:** {e}"
                        print(result)

                # ▶▶ OLLAMA RETRY ───────────────────────────────────────────────
                if result.startswith("**ERROR"):
                    print(f"   ↪ retrying {part} with local Ollama …")
                    try:
                        result = run_sync(
                            langgraph_task_specific_document_processing(
                                st.session_state.ollama_llm,   # ← local model
                                docs, query,
                                task_type=part,
                                enable_groq_TFM_check=False,   # local model ⇒ no Groq guard
                            )
                        )
                        print("   ✔ Ollama succeeded")
                    except Exception as e:
                        print("   ✖ Ollama failed:", e)
                # ───────────────────────────────────────────────────────────────

                parts_out[part] = result
                print(f"  ↳ {len(result.split())} words")
                with st.expander(f"🔍 Click to view DeepSearch {part}"):
                    st.markdown(result)


            # ── 2. combine parts into one Markdown report ───────────────────────────────
            md_chunks = []
            for part in display_order:
                if part in parts_out:                     # skip if not produced
                    md_chunks.append(f"### {part}")
                    md_chunks.append(parts_out[part])

            best_final_answer = "\n\n".join(md_chunks)

            # ── 3. If any part failed, build a fallback summary -------------------------
            #any_errors = any(text.startswith("**ERROR") for text in parts_out.values())
            
            #── 3. If any part failed, build a fallback summary -------------------------        
            all_failed = all(text.startswith("**ERROR") for text in parts_out.values())


            if all_failed:
                with st.spinner('Generating backup DeepSearch Summary (fallback option)'):
                    print("\n⚠️  One or more sections failed → generating backup summary")

                    def try_summary(llm_obj, label):
                        try:
                            print(f"→ summary via {label}")
                            return asyncio.run(langgraph_summarize_documents(llm_obj, docs, query))
                        except Exception as exc:
                            print(f"   {label} failed: {exc}")
                            return None

                    # first try Groq primary; if that fails, fall back to local Ollama
                    fallback = (
                        try_summary(st.session_state.llm, "Groq primary")
                        or try_summary(st.session_state.ollama_llm, "Ollama local")
                    )

                    if fallback:
                        best_final_answer = fallback
                    else:
                        best_final_answer += (
                            "\n\n---\n\n"
                            "_Backup summary generation also failed due to repeated time-outs._"
                        )
                

    # st.info(f"### DeepSearch Answer")
    # best_final_answer = asyncio.run(
    #     langgraph_summarize_documents(st.session_state.llm, docs, query)
    # )
    # st.markdown(best_final_answer)            
    # 
    # # Cleanup  best_final_answer into a coherent version
    # best_final_answer_cleaned = st.session_state.llm(f"""
    #                                                  Clean up this <INITIAL_DRAFT> into a coherent <FINAL_DETAILED_REPORT> 
    #                                                  with inline citations (clearly referenced within the text) that fousses on answering the <USER_QUERY>. 
    #                                                  
    #                                                  <GUIDELINES>
    #                                                  1. Coherently combine all the information from the <INITIAL_DRAFT> in a logical manner with respect to the <USER_QUERY>.
    #                                                  2. Use inline citations whereever possible.   
    #                                                  3. Only generate a single report enclosed within <FINAL_DETAILED_REPORT> .. </FINAL_DETAILED_REPORT>   tags. 
    #                                                  4. Be Exhaustive, but Donot repeat information.      
    #                                                                                 
    #                                                  </GUIDELINES>
    #   
    #                                                  <USER_QUERY>.
    #                                                  {query}
    #                                                  </USER_QUERY>.
    #                                                  
    #                                                 <INITIAL_DRAFT>
    #                                                    {best_final_answer}
    #                                                 </INITIAL_DRAFT>
    # 
    #                                                 <FINAL_DETAILED_REPORT>                                                  
    # 
    #                                                 """)
    # 
    # # extract best_final_answer_cleaned from witin <FINAL_DETAILED_REPORT>...</FINAL_DETAILED_REPORT>
    # best_final_answer_cleaned =    extract_answer_or_raw(best_final_answer_cleaned, pattern="FINAL_DETAILED_REPORT")


    df_paths = pd.DataFrame(all_paths_sorted)

    return {
        "final_answer": best_final_answer,
        "df_RAG_STAR": df_paths,
    }


# ╭────────────────────── legacy alias ─────────────────────────────╮
def RAG_STAR(
    query: str,
    superposition_fn: Callable[[str], Any],
    search_type: str = "DeepSearch",
    skip_baseline_answer: bool = False,
    **kwargs,
):
    """
    Backwards-compat wrapper to preserve old import style.
    """
    return run_RAGSTAR(
        query,
        superposition_fn,
        **kwargs,
        search_type=search_type,
        skip_baseline_answer=skip_baseline_answer,
    )
# ╰──────────────────────────────────────────────────────────────────╯


# ---- MCTS UCT fix: guaranteed exploration for unvisited children ----
def _ragstar_mcts_fixed_best_child(self):
    """UCT selection: prioritize unexplored children, else UCB1."""
    import math
    parent_visits = getattr(self, 'visits', 0) or 0
    EXPL_C = globals().get('EXPL_C', 0.85)
    def uct(ch):
        if getattr(ch, 'visits', 0) == 0:
            return float('inf')
        exploit = (getattr(ch, 'value', 0.0) / float(ch.visits))
        explore = EXPL_C * math.sqrt(max(0.0, math.log(max(1.0, float(parent_visits)))) / float(ch.visits))
        return exploit + explore
    return max(getattr(self, 'children', []), key=uct)

def run_RAGSTAR_streamlit_original(
    query: str,
    superposition_fn: Callable[[str], Any],
    *,
    search_type: str = "DeepSearch",
    skip_baseline_answer: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """Compatibility alias: same behavior as run_RAGSTAR (kept for legacy imports)."""
    return run_RAGSTAR(
        query,
        superposition_fn,
        search_type=search_type,
        skip_baseline_answer=skip_baseline_answer,
        **kwargs
    )

def run_search_streamlit_original(
    query: str,
    superposition_fn: Callable[[str], Any],
    *,
    search_type: str = "DeepSearch",
    skip_baseline_answer: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """Compatibility alias for older code paths; forwards to run_RAGSTAR."""
    return run_RAGSTAR(
        query,
        superposition_fn,
        search_type=search_type,
        skip_baseline_answer=skip_baseline_answer,
        **kwargs
    )
