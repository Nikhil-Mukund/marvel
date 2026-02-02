import re
import requests

from urllib.parse import urlparse , urlunparse
from langchain.schema import Document
from typing import Any, Dict, List, Union

from pathlib import Path   # only needed if you want to read/write files
from typing import List, Any
from copy import deepcopy
import re,json
from langchain_core.documents import Document



def _safe_json_extract(text: str) -> dict:
    """
    Extract first JSON object from a model response.
    Returns {} if parsing fails.
    """
    if not text:
        return {}
    # Grab the first {...} block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def dedup_paragraphs(text: str) -> str:
    """
    Return *text* with duplicate paragraphs removed (first appearance kept).
    A paragraph is any block of non-empty lines separated by ≥1 blank line.
    Leading/trailing whitespace is ignored for the purpose of comparison,
    but the original paragraph text (including its internal line-breaks)
    is preserved in the output.
    """
    cleaned_seen = set()
    unique_paragraphs = []
    # Split on one or more blank lines and keep the blank-line structure minimal
    raw_paragraphs = [p.rstrip()                      # trim right-hand spaces
                      for p in text.splitlines()]     # work line-by-line
    # Reassemble lines into blocks (paragraphs)
    paragraph, buffer = [], []
    for line in raw_paragraphs + ['']:                # sentinel '' to flush the last block
        if line.strip():                              # non-blank line
            buffer.append(line)
        else:                                         # blank line signals paragraph boundary
            if buffer:                                # finished a paragraph
                paragraph = '\n'.join(buffer)
                canonical = paragraph.strip()         # canonical form for hashing
                if canonical not in cleaned_seen:
                    cleaned_seen.add(canonical)
                    unique_paragraphs.append(paragraph)
                buffer = []                           # reset for next paragraph
    # Join paragraphs with a single blank line
    return '\n\n'.join(unique_paragraphs)


def sanitize_equation(eq_string):
    return eq_string.replace('\\', '\\\\').replace("'", "\\'")


def sanitize_equation_text(text):
    # Step 1: Remove standalone curly braces, but not those used in LaTeX commands
    text = re.sub(r'(?<!\\)\{(?![a-zA-Z])', '', text)  # Remove opening braces
    text = re.sub(r'(?<!\\)\}', '', text)  # Remove closing braces
    # Step 2: Escape backslashes that aren't already part of an escape sequence
    text = re.sub(r'(?<!\\)\\(?![\\\{\}])', r'\\\\', text)

def escape_braces(text):
    return text.replace('{', '[').replace('}', ']')


def nicely_format_latex_eqns(text):
    # Replace \( \) with $ $ for inline LaTeX
    text = text.replace(r'\(', '$').replace(r'\)', '$')
    # Replace \[ \] with $$ $$ for block LaTeX
    text = text.replace(r'\[', '$$').replace(r'\]', '$$')
    # Use regex to find all occurrences of [...] and replace with $$...$$
    text = re.sub(r'\[(.*?)\]', r'$$\1$$', text)
    return text


# Check if URL is accessible
def is_url_accessible(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        if response.status_code == 405:
            response = requests.get(url, allow_redirects=True, timeout=5)
        return 200 <= response.status_code < 400
    except requests.RequestException:
        return False

def validate_and_replace_urls(text, placeholder="<INACCESSIBLE URL>"):
    # Regular expression to find URLs in text
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z0-9$-_@.&+!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    def replace_url(match):
        url = match.group(0)
        if is_url_accessible(url):
            return url
        else:
            return placeholder
    # Replace URLs in the text using the 'replace_url' function
    return url_pattern.sub(replace_url, text)


def check_urls(urls):
    results = {}
    for url in urls:
        is_accessible = is_url_accessible(url)
        results[url] = "Accessible" if is_accessible else "Not accessible"
    return results



'''
def replace_plus_in_url_paths(text):
    """
    Finds all URLs in the text and replaces any '+' in the path portion
    with '%20'. Leaves query strings and other parts alone.
    """
    # Simple regex to capture most http/https URLs (stops at whitespace)
    url_pattern = re.compile(r'https?://[^\s]+')
    def replacer(match):
        original_url = match.group(0)
        parsed = urlparse(original_url)
        # Replace '+' with '%20' in the path segment
        new_path = parsed.path.replace('+', '%20')
        # Rebuild the URL with the updated path
        sanitized_url = urlunparse((
            parsed.scheme,
            parsed.netloc,
            new_path,
            parsed.params,
            parsed.query,
            parsed.fragment
        ))
        return sanitized_url
    return url_pattern.sub(replacer, text)
    '''



# ─── one compact pattern: “http://… ” or “https://… ” up to whitespace/′"′/′<′ ──
_URL_RE = re.compile(r"https?://[^\s\"\'<>,.;:)]+")

def replace_plus_in_url_paths(text: str) -> str:
    """
    Replace raw '+' characters that appear *inside the path segment*
    of well‑formed http/https URLs with '%2B'.
    • Query strings, anchors, etc. are left unchanged.
    • Malformed URLs (e.g. stray IPv6 literals) are ignored – the original
      substring is returned so the caller never crashes.
    """
    def _patch(match: re.Match) -> str:
        url = match.group(0)
        # Try to parse; bail out safely on any ValueError
        try:
            scheme, netloc, path, params, query, frag = urlparse(url)
        except ValueError:          # “Invalid IPv6 URL” and friends
            return url              # keep original, no exception
        if "+" not in path:         # nothing to fix
            return url
        path = path.replace("+", "%2B")      #   ‘+’ → “%2B”
        return urlunparse((scheme, netloc, path, params, query, frag))
    return _URL_RE.sub(_patch, text)


def add_color_filename_tags(text):
    pattern = r'<<\s*filename:\s*(.*?)\s*>>'
    replacement = r':red[<< filename: \1 >>]'
    return re.sub(pattern, replacement, text)


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





def wrap_urls_in_angle_brackets(text: str) -> str:
    """
    Wraps URLs starting with 'http://' or 'https://' in angle brackets '<' and '>'.
    It handles URLs that are:
    - Enclosed within single quotes inside a larger string.
    - The entire input string is a URL.
    Args:
        text (str): The input text containing URLs.
    Returns:
        str: The text with detected URLs wrapped in angle brackets.
    """
    # Pattern to match URLs within single quotes
    single_quote_pattern = r"'(?P<url>https?://[^'<>]+)'"    
    # Function to wrap matched URLs with angle brackets
    def replace_single_quote_match(match: re.Match) -> str:
        url = match.group('url')
        # Avoid double-wrapping if already in angle brackets
        if url.startswith('<') and url.endswith('>'):
            return match.group(0)
        else:
            return f"'<{url}>'"
    # First, replace URLs within single quotes
    updated_text = re.sub(single_quote_pattern, replace_single_quote_match, text)
    # Pattern to match the entire string as a URL
    entire_url_pattern = r"^(https?://[^<>]+)$"
    # Check if the entire string is a URL
    match_entire = re.match(entire_url_pattern, updated_text)
    if match_entire:
        url = match_entire.group(1)
        # Avoid double-wrapping if already in angle brackets
        if not (url.startswith('<') and url.endswith('>')):
            updated_text = f"<{url}>"    
    return updated_text





def wrap_urls_in_metadata(documents: List[Document], fields: List[str] = ['source', 'url']) -> None:
    """
    Iterates through a list of Document objects and wraps URLs in specified metadata fields
    with angle brackets if they start with 'http://' or 'https://'.
    Args:
        documents (List[Document]): A list of LangChain Document objects to be processed.
        fields (List[str], optional): Metadata fields to inspect for URLs. Defaults to ['source', 'url'].

    Returns:
        None. The function modifies the Document objects in-place.
    """
    # Regular expression pattern to match URLs starting with http:// or https://
    url_pattern = re.compile(r'^https?://[^\s<>"]+$')
    def process_value(value: Any) -> Any:
        """
        Recursively processes a value to wrap URLs in angle brackets.

        Args:
            value (Any): The value to process (can be str, dict, list, etc.).

        Returns:
            Any: The processed value with URLs wrapped in angle brackets.
        """
        if isinstance(value, str):
            # Check if the string is a URL starting with http:// or https://
            if url_pattern.match(value):
                # Check if already wrapped in angle brackets
                if not (value.startswith('<') and value.endswith('>')):
                    return f"<{value}>"
            return value  # Return unchanged if not a URL or already wrapped
        elif isinstance(value, dict):
            # Recursively process each key-value pair in the dictionary
            return {k: process_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            # Recursively process each item in the list
            return [process_value(item) for item in value]
        else:
            # For other data types, return the value unchanged
            return value
    for idx, doc in enumerate(documents, start=1):
        print(f"\nProcessing Document {idx}: doc_id = {doc.metadata.get('doc_id', 'N/A')}")
        for field in fields:
            if field in doc.metadata:
                original_value = doc.metadata[field]
                print(f"  Original '{field}': {original_value}")                
                # Process the value (handles str, dict, list)
                updated_value = process_value(original_value)                
                # Update the metadata only if changes were made
                if updated_value != original_value:
                    doc.metadata[field] = updated_value
                    print(f"  Updated '{field}': {updated_value}")
                else:
                    print(f"  '{field}' is already properly formatted or not a target URL.")
            else:
                print(f"  Field '{field}' not found in metadata.")


# code to replace arxiv id in /RAG_DataSets/LatexData/downloaded_arxiv_articles with actual PDF URLS
def update_LatexData_downloaded_arxiv_articles_id_with_PDF_URL(input):
    pattern = r"\.\/RAG_DataSets\/LatexData\/downloaded_arxiv_articles\/([^\/]+)\/[^\/]+\.tex"
    replacement = r"https://www.arxiv.org/pdf/\1.pdf"
    output = re.sub(pattern, replacement, input)
    return output



def detect_and_convert_tavily_ligo_public_urls(original_url):
    """
    Extracts the LIGO code from the original URL and constructs the new URL.    
    Parameters:
        original_url (str): The original URL containing the LIGO code.        
    Returns:
        str: The newly constructed URL, or None if the pattern is not found
             or if 'ligo.org' is not present in the original URL.
    """
    print(f"Original URL: {original_url}")  # Debug: Print the original URL    
    # Check if 'ligo.org' is present in the URL
    if 'ligo.org' not in original_url:
        print("Ignored: URL does not contain 'ligo.org'.")  # Debug: Indicate ignored URL
        return original_url
    # Regex pattern to match 1 letter followed by 7 digits
    pattern = r'\b([A-Za-z]\d{7})\b'    
    # Search for the pattern in the original URL
    match = re.search(pattern, original_url)    
    if match:
        code = match.group(1)
        print(f"Matched Code: {code}")  # Debug: Print the matched code
        
        # Construct the new URL using the extracted code
        new_url = f'https://dcc.ligo.org/LIGO-{code}/public'
        print(f"New URL: {new_url}")  # Debug: Print the new URL
        return new_url
    else:
        print("No matching LIGO code found.")  # Debug: Indicate no match
        return original_url




def detect_and_convert_public_urls(original_url, domain_keywords=None):
    """
    Convert a URL that contains a LIGO-style document code (e.g. “L2400123”)
    to its canonical public DCC link.

    Parameters
    ----------
    original_url : str
        The URL to examine.
    domain_keywords : str, iterable, or None, default "ligo.org"
        • str      – the substring that must appear in the URL  
        • iterable   – any one of these substrings must appear  
        • None      – skip the domain check entirely

    Returns
    -------
    str
        The converted URL if we find a code (and the domain check passes);
        otherwise the original URL.
    """
    # --- Optional domain gating --------------------------------------------------
    if domain_keywords is not None:
        # Normalise to a tuple of keywords
        if isinstance(domain_keywords, str):
            keywords = (domain_keywords,)
        else:
            keywords = tuple(domain_keywords)

        # Exit early if none of the keywords are present
        if not any(k in original_url for k in keywords):
            return original_url

    # --- Detect a one-letter, seven-digit LIGO code ------------------------------
    match = re.search(r"\b([A-Za-z]\d{7})\b", original_url)
    if not match:
        return original_url

    code = match.group(1)
    return f"https://dcc.ligo.org/LIGO-{code}/public"



def hyperlink_dcc_urls(text):
    pattern = r'<(dcc\.ligo\.org/SOMETHING)>'
    replacement = r'<https://\1>'
    return re.sub(pattern, replacement, text)

def normalize_quotes(s: str) -> str:
    s = s.strip()
    # Use a regex to remove matching outer quotes (either single or double)
    m = re.match(r"^(['\"])(.*)\1$", s)
    if m:
        inner = m.group(2)
    else:
        inner = s
    return f"'{inner}'"




def extract_answer_or_raw(text, pattern="FINAL_DETAILED_REPORT"):
    """
    Return the first <PATTERN>…</PATTERN> (with optional spaces inside < >),
    sans tags. If missing or empty, return the original text.
    """
    # Allow spaces after '<', before '>', and around the tag name.
    tag = re.escape(pattern)
    answer_pat = re.compile(
        rf"<\s*{tag}\s*>(.*?)</\s*{tag}\s*>",
        re.DOTALL | re.IGNORECASE
    )
    text = dedup_paragraphs(text)  # your helper
    m = answer_pat.search(text)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return text






def latexify_documents(docs: List[Any]) -> List[Document]:
    """
    Return *new* Documents whose ``page_content`` strings have LaTeX-looking
    fragments wrapped in proper delimiters so that Streamlit/MathJax renders
    them nicely.

    • Any token containing typical LaTeX commands (\\left, \\frac, \\sqrt,
      matrix environments, etc.) is detected.
    • If the fragment is **already** inside balanced $…$ or $$…$$ delimiters
      it is left unchanged.
    • Otherwise it is cleaned (duplicate commas, = =, bad “\\left($” sequences
      fixed) and wrapped:
        – long / multi-line / environment fragments → `$$ … $$`
        – everything else → `$ … $`
    • Non-Document items in the input list are returned untouched.

    Parameters
    ----------
    docs : List[Any]
        Heterogeneous list that should contain LangChain Document objects.
    Returns
    -------
    List[Document]
        New list with the same ordering; each Document is a deep-copy of the
        original with its ``page_content`` modified.
    """
    # ── regex helpers kept local to avoid polluting global namespace ──
    LATEX_CMD_RE = re.compile(
        r"\\left\\?\(|\\right\\?\)|\\frac|\\sqrt|\\begin\{|\\end\{|\\matrix|\\theta"
    )
    BAL_INLINE = re.compile(r"\$(?!\$).*?\\.*?\$(?!\$)", re.DOTALL)
    BAL_BLOCK  = re.compile(r"\$\$.*?\\.*?\$\$", re.DOTALL)
    def looks_like_latex(s: str) -> bool:
        return bool(LATEX_CMD_RE.search(s))
    def already_balanced(s: str) -> bool:
        return bool(BAL_INLINE.search(s) or BAL_BLOCK.search(s))
    def clean(tex: str) -> str:
        tex = re.sub(r",[ ,]+", ", ", tex)                 # ", ,", ",  ,"
        tex = re.sub(r"= ?= ?", "=", tex)                  # "= =" → "="
        tex = tex.replace(r"\left($", r"\left(")           # "\left($" → "\left("
        tex = tex.replace(r"$\right)", r"\right)")         # "$\right)" → "\right)"
        tex = tex.replace("$)", ")")
        tex = re.sub(r"\s{2,}", " ", tex)                  # collapse spaces
        return tex.strip()
    def wrap(tex: str) -> str:
        tex = clean(tex)
        if "\n" in tex or len(tex) > 120 or r"\begin" in tex:
            return f"$$\n{tex}\n$$"
        return f"${tex}$"
    # ── main loop ────────────────────────────────────────────────────
    fixed: List[Document] = []
    for orig in docs:
        # leave non-Document items unchanged
        if not (hasattr(orig, "page_content") and hasattr(orig, "metadata")):
            fixed.append(orig)
            continue
        d = deepcopy(orig)
        parts = []
        tokens = re.split(r"(\s+)", d.page_content)  # keep whitespace tokens
        for tok in tokens:
            if looks_like_latex(tok) and not already_balanced(tok):
                parts.append(wrap(tok))
            else:
                parts.append(tok)
        d.page_content = "".join(parts)
        fixed.append(d)
    return fixed

def _pp_output_to_text(x) -> str:
    """Coerce LLM JSON 'output' field into a displayable string safely."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x

    # If the model returns a list (common for links)
    if isinstance(x, list):
        # list[dict] => try to format nicely
        if x and all(isinstance(i, dict) for i in x):
            lines = []
            for i in x:
                url = (i.get("url") or i.get("link") or i.get("href") or "").strip()
                note = (i.get("note") or i.get("title") or i.get("desc") or "").strip()
                if url and note:
                    lines.append(f"- {url} — {note}")
                elif url:
                    lines.append(f"- {url}")
                else:
                    lines.append(f"- {json.dumps(i, ensure_ascii=False)}")
            return "\n".join(lines)
        # list[str] or mixed
        return "\n".join(f"- {str(i)}" for i in x)

    # dict => pretty JSON
    if isinstance(x, dict):
        return json.dumps(x, ensure_ascii=False, indent=2)

    return str(x)