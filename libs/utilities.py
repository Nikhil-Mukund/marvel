import csv
import os
import time, threading, re


from config import config



# ╭────────────────────────── configuration ─────────────────────────╮
CFG = config.load_config()

# MCTS parameters
MCTS_CFG = CFG["retrieval"]

def estimate_tokens_from_text(text):
    """
    Estimate the number of tokens in the given text using simple heuristics.

    Parameters
    ----------
    text : str
        The input text for which we want to estimate the number of tokens.

    Returns
    -------
    int
        An approximate token count.

    Notes
    -----
    This is a rough estimation and not an exact token count. 
    For exact token counts, use the actual tokenizer of the language model.
    """
    # Strip leading/trailing whitespace and normalize spaces
    cleaned_text = text.strip()
    
    # Split on whitespace to count words
    words = cleaned_text.split()
    word_count = len(words)
    
    # Count characters (including spaces and punctuation)
    character_count = len(cleaned_text)
    
    # Approximation 1 (Word-based): 
    # 75 words ~ 100 tokens => 1 word ~ 1.333 tokens
    # tokens ≈ word_count * (4/3)
    tokens_est_from_words = word_count * (4/3)
    
    # Approximation 2 (Character-based):
    # 1 token ~ 4 characters
    # tokens ≈ character_count / 4
    if character_count > 0:
        tokens_est_from_chars = character_count / 4
    else:
        tokens_est_from_chars = 0
    
    # Combine the two estimates by averaging them for a slightly more robust estimate
    estimated_tokens = (tokens_est_from_words + tokens_est_from_chars) / 2
    
    # Round to an integer for a cleaner estimate
    return int(round(estimated_tokens))




def load_acronyms(csv_file_path):
    """
    Load acronyms from the given CSV file into a dictionary.
    CSV must have headers: ACRONYM, DEFINITION.
    Returns a dictionary where keys are acronyms and values are definitions.
    """
    acronyms_dict = {}
    csv_file_path_mod  = f"./{csv_file_path}"
    with open(csv_file_path_mod, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = row['ACRONYM'].strip()
            value = row['DEFINITION'].strip()
            acronyms_dict[key] = value
    return acronyms_dict

def get_acronym_definition(acronym, acronyms_dict):
    """
    Return the definition for the given acronym from the pre-loaded dictionary.
    If not found, returns None.
    """
    return acronyms_dict.get(acronym, None)


# rate_limit_light  ────────────────────────────────────────────────


TPM_LIMIT = MCTS_CFG["GROQ_MAX_TOKEN_PER_MINUTE"]      # your Groq model’s limit

_word_pat = re.compile(r"\S+")

_lock           = threading.Lock()
window_start_ts = 0.0
tokens_this_min = 0

def approx_token_count(text: str) -> int:
    """Roughly convert word count to tokens (words * 0.75)."""
    words = len(_word_pat.findall(text))
    return int(words * 0.75)

def tpm_gate(tokens_needed: int) -> None:
    """Block until sending `tokens_needed` keeps us under the 60-s TPM window."""
    global tokens_this_min, window_start_ts
    with _lock:
        now = time.time()
        if now - window_start_ts >= 60:
            window_start_ts = now
            tokens_this_min = 0
        if tokens_this_min + tokens_needed <= TPM_LIMIT:
            tokens_this_min += tokens_needed
            return
        wait = 60 - (now - window_start_ts) + 0.1
    print(f"[TPM-gate] need {tokens_needed} tok (limit {TPM_LIMIT}); sleep {wait:.1f}s")
    time.sleep(wait)
    tpm_gate(tokens_needed)
