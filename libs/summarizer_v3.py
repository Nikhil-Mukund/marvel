import asyncio
import re
from typing import List, Literal, TypedDict

from langchain_core.runnables.config import RunnableConfig

from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain.schema.runnable import RunnableLambda
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from libs.regex import nicely_format_latex_eqns, dedup_paragraphs , extract_answer_or_raw

from libs.retrievers import ensembele_superposition_answer_from_docs

from libs.utilities import approx_token_count as tok_cnt


# ─── parameters you can tweak ────────────────────────────────────────
TOKENS_PER_CHUNK = 800        # soft limit per slice
WORDS_PER_TOKEN  = 1.3        # ≈ English words / OpenAI-style token

# very light estimator (no tiktoken)
def approx_tokens(txt: str) -> int:
    return int(len(txt.split()) / WORDS_PER_TOKEN)

def split_to_chunks(text: str, meta: str, limit: int) -> list[str]:
    """Return a list of ≤limit-token slices, each with the same metadata tail."""
    words      = text.split()
    step       = int(limit * WORDS_PER_TOKEN)
    slices     = [
        " ".join(words[i : i + step]) + f"\n\n{meta}"
        for i in range(0, len(words), step)
    ]
    return slices

# --- ainvoke watchdog: verbose progress + optional hard timeout ---
import os, time, asyncio, contextlib, traceback

# Per-step timeout (seconds); set to "0" or leave unset to allow infinite wait.
SUMMARIZER_STEP_TIMEOUT_SEC=30 
SUMMARIZER_HEARTBEAT_SEC=10 
_STEP_TIMEOUT = float(SUMMARIZER_STEP_TIMEOUT_SEC)  # e.g., 120
_HEARTBEAT = float(SUMMARIZER_HEARTBEAT_SEC)       # prints every 10s
GROQ_MAX_CONCURRENCY=8

async def ainvoke_with_watchdog(runnable, inputs, config, label="refine"):
    """
    Wraps runnable.ainvoke(inputs, config) so we get:
      - start/end timestamps,
      - heartbeat prints during long waits,
      - optional hard timeout via asyncio.wait_for.
    """
    t0 = time.monotonic()
    print(f"[{label}] ainvoke START (timeout={_STEP_TIMEOUT or '∞'}s, heartbeat={_HEARTBEAT}s)")
    task = asyncio.create_task(runnable.ainvoke(inputs, config))

    try:
        if _STEP_TIMEOUT > 0:
            # Wait with a timeout, but still print heartbeats while pending
            deadline = t0 + _STEP_TIMEOUT
            while True:
                now = time.monotonic()
                remaining = max(0.0, deadline - now)
                if remaining == 0.0:
                    raise asyncio.TimeoutError(f"{label} timed out after {_STEP_TIMEOUT:.0f}s")

                # Wait in short slices so we can print a heartbeat
                slice_sec = min(_HEARTBEAT, remaining)
                done, _ = await asyncio.wait({task}, timeout=slice_sec)
                if task in done:
                    result = task.result()
                    dt = time.monotonic() - t0
                    print(f"[{label}] ainvoke DONE in {dt:.2f}s")
                    return result
                else:
                    elapsed = time.monotonic() - t0
                    print(f"[{label}] still waiting… {elapsed:.1f}s elapsed")
        else:
            # No timeout: just heartbeat
            while True:
                done, _ = await asyncio.wait({task}, timeout=_HEARTBEAT)
                if task in done:
                    result = task.result()
                    dt = time.monotonic() - t0
                    print(f"[{label}] ainvoke DONE in {dt:.2f}s")
                    return result
                else:
                    elapsed = time.monotonic() - t0
                    print(f"[{label}] still waiting… {elapsed:.1f}s elapsed")
    except asyncio.TimeoutError as e:
        print(f"[{label}] ainvoke TIMEOUT: {e}")
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        raise
    except Exception as e:
        print(f"[{label}] ainvoke ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise
# -------------------------------------------------------------------

# --- limit concurrent Groq calls to avoid pool starvation ---

_GROQ_MAX_CONCURRENCY = int(GROQ_MAX_CONCURRENCY)
# ---- concurrency gate to prevent pool starvation (Python 3.9-safe) ----
import os, asyncio
from typing import Optional

# Define the global at module import so NameError won't occur later.
_GROQ_SEM: Optional[asyncio.Semaphore] = None

def _groq_sem() -> asyncio.Semaphore:
    """
    Lazily create the semaphore the *first time* it's needed, when a loop exists.
    Must be called from within an async context (i.e., when a loop is running).
    """
    global _GROQ_SEM
    if _GROQ_SEM is None:
        # This line executes inside your coroutine (a running loop exists),
        # so creating the Semaphore here is safe in Streamlit/Python 3.9.
        _GROQ_SEM = asyncio.Semaphore(_GROQ_MAX_CONCURRENCY)
    return _GROQ_SEM
# -----------------------------------------------------------------------

# ------------------------------------------------------------

###############################################################################
# Define the data structure for state
###############################################################################
class State(TypedDict):
    contents: List[str]
    index: int
    summary: str


###############################################################################
# Function that builds the summarization graph and runs it
###############################################################################
async def langgraph_summarize_documents(llm, documents: List[Document],user_query) -> str:
    """
    Summarize the given documents using a multi-step refinement approach.
    
    :param llm: A LangChain-compatible LLM, e.g. Ollama(...) or similar
    :param documents: A list of LangChain Document objects
    :return: The final summary (str)
    """

    # -------------------------------------------------------------------------
    # 1. Define Prompt Templates
    # -------------------------------------------------------------------------

    refine_template = """
    <GUIDELINES>
    1. Retain all the current information from existing <EXISTING_REPORT> that answers the QUERY and merge the <ADDITIONAL_REPORT> to create the <FINAL_DETAILED_REPORT>
    2. Remember to retain all the information. Donot skip citations, urls and other key information from  both the <EXISTING_REPORT> and <ADDITIONAL_REPORT>
    3. Maintain the same style as the existing <EXISTING_REPORT>
    4. Use <FINAL_DETAILED_REPORT>...</FINAL_DETAILED_REPORT> tags for the FINAL_DETAILED_REPORT
    5. Only generate the <New FINAL_DETAILED_REPORT> and nothing else. Donot explain the rationale etc.
    </GUIDELINES>

    <QUERY>
    {user_query}   
    </QUERY>

    <EXISTING_REPORT>
    {existing_answer}
    </EXISTING_REPORT>

    <NEW_ADDITIONAL_REPORT>
    ------------
    {context}
    ------------
    </NEW_ADDITIONAL_REPORT>

    <FINAL_DETAILED_REPORT>
    """
    refine_prompt = ChatPromptTemplate([("human", refine_template)])
    
    # -------------------------------------------------------------------------
    # 2. Define the pipeline (chains)
    # -------------------------------------------------------------------------
    #initial_summary_chain = summarize_prompt | llm | StrOutputParser()
    #initial_summary_chain = asyncio.run(run_langgraph_summarization(llm,documents,user_query))   
    #initial_summary_chain = langgraph_summarize_documents(llm, documents,user_query)

    FINAL_RESPONSE, QA_ANSWER, WIKI_SUMMARY, MORE_DETAILS,filtered_retrieved_docs_LLM_answers,QA_ANSWER_AGGREGATOR = ensembele_superposition_answer_from_docs(documents,user_query)            
    # get baseline answer
    initial_summary_chain = RunnableLambda(
    # the callable receives the usual input dict but we ignore it here
    lambda _: QA_ANSWER + "\n\n" + WIKI_SUMMARY
    )   
    
    #refine_summary_chain  = refine_prompt    | llm | StrOutputParser()

    # ── link it into your chain ───────────────────────────────────────────────
    refine_summary_chain = (
        refine_prompt        # -> Prompt template
        | llm                # -> Raw LLM string
        | StrOutputParser()  # -> Plain string
        | RunnableLambda(extract_answer_or_raw)  # -> Post-processed string
)


    # -------------------------------------------------------------------------
    # 3. Define the node functions
    # -------------------------------------------------------------------------
    async def generate_initial_summary(state: State, config: RunnableConfig):
        summary = await initial_summary_chain.ainvoke(
                {} , #{"context": state["contents"][0]},
                config,  # Pass the customized config here
        )
        return {"summary": summary, "index": 1}

    async def refine_summary(state: State, config: RunnableConfig):
        content = state["contents"][state["index"]]

        # Wait if token exceeds the max tokens per limit of groq
        print("TPM check DISABLED in summarizer_v2; no waiting ")
        prompt_tokens = tok_cnt(state["summary"]) + tok_cnt(content) + 50
        print("[summarizer_v2] TPM gating disabled; proceeding without wait (prompt_tokens computed).")

        summary = await refine_summary_chain.ainvoke(
            {"user_query":user_query,"existing_answer": state["summary"], "context": content},
            config,  # Pass the customized config here
        )
        return {"summary": summary, "index": state["index"] + 1}

    def should_refine(state: State) -> Literal["refine_summary", END]:
        """Decide whether to refine the summary or end."""
        if state["index"] >= len(state["contents"]):
            return END
        else:
            return "refine_summary"

    # -------------------------------------------------------------------------
    # 4. Build the StateGraph
    # -------------------------------------------------------------------------
    graph = StateGraph(State)
    graph.add_node("generate_initial_summary", generate_initial_summary)
    graph.add_node("refine_summary", refine_summary)

    graph.add_edge(START, "generate_initial_summary")
    graph.add_conditional_edges("generate_initial_summary", should_refine)
    graph.add_conditional_edges("refine_summary", should_refine)

    app = graph.compile()

    # -------------------------------------------------------------------------
    # 5. Create a RunnableConfig with increased recursion limit
    # -------------------------------------------------------------------------
    config = RunnableConfig(recursion_limit=1000)  # Set to desired limit

    # -------------------------------------------------------------------------
    # 6. Run the graph and capture the final summary
    # -------------------------------------------------------------------------
    # ----- build safe "contents" list ---------------------------------
    contents = []
    for d in documents:
        if hasattr(d, "page_content") and hasattr(d, "metadata"):
            contents.append(f"{d.page_content}\n\n{d.metadata}")
        else:                     # fallback for ellipsis, strings, dicts, None…
            contents.append(str(d))
    # optional: skip the summariser when there is literally nothing useful
    if all(c.strip() == "" for c in contents):
        last_summary = ""
    else:
        # ----- run the async summariser -----------------------------------
        last_summary = ""
        async for step in app.astream(
            {"contents": contents},
            stream_mode="values",
            config=config,
        ):
            if step.get("summary"):
                last_summary = step["summary"]

    return last_summary



###############################################################################
# Function that builds the uses langgraph graph to extract task specific info from a bunch of reports
###############################################################################
async def langgraph_task_specific_document_processing(llm, documents: List[Document],user_query,task_type,enable_groq_TFM_check) -> str:
    """
    Function that builds the uses langgraph graph to extract task specific info from a bunch of reports
    
    :param llm: A LangChain-compatible LLM, e.g. Ollama(...) or similar
    :param documents: A list of LangChain Document objects
    :return: The final task report (str)
    """

    # -------------------------------------------------------------------------
    # 1. Define Prompt Templates
    # -------------------------------------------------------------------------

    refine_template = """
    <GUIDELINES>
    0. If any sentences contain inline citation markers of the form [@"..."] (e.g., [@"url:https://..."]), preserve them verbatim and keep them attached to the sentence. Do not invent, drop, or renumber citations. When combining or reordering sentences, move their exact markers with the corresponding sentences; do not aggregate or split markers unless the sentence is split/joined.
    1. Given a <TASK_TYPE>, Retain all the current information from existing <EXISTING_REPORT> that answers the QUERY and merge info from <ADDITIONAL_REPORT> and generate the <TASK_TYPE> specific <TASK_OUTPUT>
    2. The <TASK_TYPE> would be to generate just one of the following : <INTRODUCTION>,<BACKGROUND>,<KEY_FACTS>,<FINER_DETAILS>,<METHODOLOGY>,<CODES>,<TABLES>,<UNEXPECTED_DETAIL>,<SUMMARY>,<CONCLUSIONS>,<REFERENCES>
    3. ALways remember to adhere to the requested <TASK_TYPE> and nothing else. 
    4. Always remember to reflect and refine the existing info based on the new info. Make sure to have a coherent narration.
    5. Always remember to only use information from <EXISTING_REPORT> and <NEW_ADDITIONAL_REPORT>. Donot hallucinate.
    6. If ur are asked to generate <INTRODUCTION> only generate & refine <INTRODUCTION> .  Donot generate citations for <INTRODUCTION>. 
    7. If ur are asked to generate <BACKGROUND> , Using the context and inline citations, bring in the general motivation for the topic and Explain the key terms and concepts.
    8. If ur are asked to generate <KEY_FACTS> only generate & refine <KEY_POINTS> as bulleted list. Only use inline citations for <KEY_FACTS>. Donot generate extra source citations.
    9. If ur are asked to generate <FINER_DETAILS> only generate & refine <FINER_DETAILS> which contains all the technical details, including the not-so-obvious and rather unexpected points if any from the context. Use inline citations. 
    10. If ur are asked to generate <REFERENCES> only generate & refine <REFERENCES> as a bulleted list.  Always remember to not repeat the same citations.
    11. Use <TASK_OUTPUT>...</TASK_OUTPUT> tags for the TASK_OUTPUT
    12. Only generate the <TASK_OUTPUT> and nothing else. Donot explain the rationale etc.
    </GUIDELINES>

    <QUERY>
    {user_query}   
    </QUERY>

    <TASK_TYPE>
    {task_type}   
    </TASK_TYPE>

    <EXISTING_REPORT>
    {existing_answer}
    </EXISTING_REPORT>

    <NEW_ADDITIONAL_REPORT>
    ------------
    {context}
    ------------
    </NEW_ADDITIONAL_REPORT>

    <TASK_OUTPUT>
    """
    refine_prompt = ChatPromptTemplate([("human", refine_template)])
    
    # -------------------------------------------------------------------------
    # 2. Define the pipeline (chains)
    # -------------------------------------------------------------------------


    FINAL_RESPONSE, QA_ANSWER, WIKI_SUMMARY, MORE_DETAILS,filtered_retrieved_docs_LLM_answers,QA_ANSWER_AGGREGATOR = ensembele_superposition_answer_from_docs(documents,user_query)            
    # get baseline answer
    initial_summary_chain = RunnableLambda(
    # the callable receives the usual input dict but we ignore it here
    lambda _: QA_ANSWER + "\n\n" + WIKI_SUMMARY
    )   
    
    #refine_summary_chain  = refine_prompt    | llm | StrOutputParser()

    # ── link it into your chain ───────────────────────────────────────────────
    refine_summary_chain = (
        refine_prompt        # -> Prompt template
        | llm                # -> Raw LLM string
        | StrOutputParser()  # -> Plain string
        | RunnableLambda(extract_answer_or_raw)  # -> Post-processed string
)


    # -------------------------------------------------------------------------
    # 3. Define the node functions
    # -------------------------------------------------------------------------
    async def generate_initial_summary(state: State, config: RunnableConfig):
        summary = await initial_summary_chain.ainvoke({}, config)
        return {"summary": summary, "index": 1}

    async def refine_summary(state: State, config: RunnableConfig):
        """Refine by merging the next content chunk.  Any LLM / chain error is
        logged and re-raised so LangGraph doesn’t silently loop forever."""
        idx      = state["index"]
        content  = state["contents"][idx]
        print(f"[refine] idx {idx} / {len(state['contents'])-1}")   # ← add        

        try:

            # Wait if token exceeds the max tokens per limit of groq
            if enable_groq_TFM_check:
                print("TPM check DISABLED in summarizer_v2; no waiting ")
                prompt_tokens = approx_tokens(state["summary"]) + approx_tokens(content) + 50
                if enable_groq_TFM_check:
                    print("[summarizer_v2] TPM gating disabled; proceeding without wait (prompt_tokens computed).")

            async with _groq_sem():
                summary = await ainvoke_with_watchdog(
                    refine_summary_chain,
                    {
                        "user_query": user_query,
                        "existing_answer": state["summary"],
                        "context": content,
                        "task_type": task_type,
                    },
                    config,
                    label="refine"
                )

        except Exception as e:
            print(f"⚠️ refine_summary failed on chunk {idx} → {e}")
            raise                                    # re-raise so you see the stack-trace

        return {"summary": summary, "index": idx + 1}

    def should_refine(state: State) -> Literal["refine_summary", END]:
        return END if state["index"] >= len(state["contents"]) else "refine_summary"


    # -------------------------------------------------------------------------
    # 4. Build the StateGraph
    # -------------------------------------------------------------------------
    graph = StateGraph(State)
    graph.add_node("generate_initial_summary", generate_initial_summary)
    graph.add_node("refine_summary", refine_summary)

    graph.add_edge(START, "generate_initial_summary")
    graph.add_conditional_edges("generate_initial_summary", should_refine)
    graph.add_conditional_edges("refine_summary", should_refine)

    app = graph.compile()

    # -------------------------------------------------------------------------
    # 5. Create a RunnableConfig with increased recursion limit
    # -------------------------------------------------------------------------
    config = RunnableConfig(recursion_limit=1000)  # Set to desired limit

    # -------------------------------------------------------------------------
    # 6. Run the graph and capture the final summary
    # -------------------------------------------------------------------------
    # ----- build safe "contents" list ---------------------------------
    contents = []
    for d in documents:
        if hasattr(d, "page_content") and hasattr(d, "metadata"):
            chunks = split_to_chunks(d.page_content, str(d.metadata), TOKENS_PER_CHUNK)
            contents.extend(chunks)
        else:
            chunks = split_to_chunks(str(d), "", TOKENS_PER_CHUNK)
            contents.extend(chunks)

    # optional: skip the summariser when there is literally nothing useful
    if all(c.strip() == "" for c in contents):
        last_summary = ""
    else:
        # ----- run the async summariser -----------------------------------
        last_summary = ""
        async for step in app.astream(
            {"contents": contents},
            stream_mode="values",
            config=config,
        ):
            if step.get("summary"):
                last_summary = step["summary"]

    return last_summary