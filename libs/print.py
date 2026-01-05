from typing import Any, List
import html
import json


def print_wide_line_message(message:str):
    print(f'\n#####################################\n{message}\n#####################################\n')



#def pretty_print_docs(docs: List[Any], print_output: bool = True) -> str:
#    """
#    Safely format a heterogeneous list of LangChain Documents (or anything else)
#    into Markdown for Streamlit.
#
#    • Objects with .page_content + .metadata are rendered normally.
#    • Everything else is stringified and prefixed with “(non-Document)”.
#    """
#    separator = "\n\n---\n\n"
#    snippets: List[str] = []
#    for idx, d in enumerate(docs):
#        if hasattr(d, "page_content") and hasattr(d, "metadata"):
#            header   = f"**Document {idx+1}:**"
#            content  = str(d.page_content)
#            meta     = f"*Metadata:* {getattr(d, 'metadata', {})}"
#            snippet  = f"{header}\n\n{content}\n\n{meta}"
#        else:
#            # fallback for ellipsis, strings, dicts, None, etc.
#            snippet  = f"**Document {idx+1} (non-Document):**\n\n{str(d)}"
#            # optional: log a warning
#            print(f"[pretty_print_docs] ⚠️ skipped non-Document at position {idx} → {type(d)}")
#        snippets.append(snippet)
#    output = separator.join(snippets)
#    if print_output:
#        print(output)
#    return output
#




def pretty_print_docs(
    docs: List[Any],
    *,
    print_output: bool = True,
    escape_metadata: bool = True,
    show_in_streamlit: bool = False,
) -> str:
    try:
        import streamlit as st
    except ImportError:
        st = None
        show_in_streamlit = False

    sep = "\n\n---\n\n"
    snippets: List[str] = []

    def _safe_meta(meta_obj: Any) -> str:
        txt = json.dumps(meta_obj, ensure_ascii=False, indent=2, default=str)  # ← patched
        return html.escape(txt) if escape_metadata else txt
    for idx, d in enumerate(docs):
        if hasattr(d, "page_content") and hasattr(d, "metadata"):
            header  = f"**Document {idx+1}:**"
            content = str(d.page_content)
            meta    = f"<pre>{_safe_meta(d.metadata)}</pre>"
            snippet = f"{header}\n\n{content}\n\n{meta}"
        else:
            snippet = (
                f"**Document {idx+1} (non-Document):**\n\n"
                f"{html.escape(str(d)) if escape_metadata else str(d)}"
            )
        snippets.append(snippet)

    out = sep.join(snippets)

    if print_output:
        print(out)
    if show_in_streamlit and st is not None:
        st.markdown(out, unsafe_allow_html=True)
    return out





def streamlit_add_msg(st,role="assistant",message="",persist=False):
    st.markdown(message,unsafe_allow_html=True)
    if  persist:
        st.session_state.chat_history.append({"role": role, "message": message})


def streamlit_add_line(st,role="assistant-0",message="\n-----\n",persist=False):
    st.markdown("\n-----\n")
    if  persist:    
        st.session_state.chat_history.append({"role": role, "message": message})


def streamlit_add_bold_heading(st,role="assistant-0",message="",persist=False):
    st.markdown(f"# **:green[{message}]**")
    if  persist:    
        st.session_state.chat_history.append({"role": role, "message": message})

