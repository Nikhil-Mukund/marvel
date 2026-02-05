from __future__ import annotations
import xml.etree.ElementTree as ET
from datetime import datetime
import platform
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from dateutil.parser import parse as parse_date
from urllib.parse import urljoin
from langchain.schema import Document
from config import config
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from ragatouille import RAGPretrainedModel
from typing import List
from pydantic import Field
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.docstore.document import Document
import re
import traceback
from fuzzywuzzy import fuzz
from math import ceil
from libs.faiss import load_faiss_vectorstore, get_docs_from_faiss_vectorstore
from libs.regex import nicely_format_latex_eqns, dedup_paragraphs , latexify_documents
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers import EnsembleRetriever
from langchain.chains import RetrievalQA
from libs.print import print_wide_line_message, pretty_print_docs,streamlit_add_msg,streamlit_add_line,streamlit_add_bold_heading
from libs.regex import wrap_urls_in_metadata,wrap_urls_in_angle_brackets,replace_plus_in_url_paths,update_LatexData_downloaded_arxiv_articles_id_with_PDF_URL,detect_and_convert_tavily_ligo_public_urls,detect_and_convert_public_urls,hyperlink_dcc_urls
from libs.prompts import get_prompt
import uuid
from langchain.prompts import PromptTemplate


import re
import requests
from bs4 import BeautifulSoup
from langchain.docstore.document import Document   

from pdfminer.high_level import extract_text
import requests
from io import BytesIO
import re
from typing import Any, List
from langchain_core.prompts import PromptTemplate
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any

from typing import List, Optional, Dict, Any
from pydantic import Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun  # optional, depending on version



import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


import os
from langchain_community.tools import TavilySearchResults



# DuckDuckGo Streamlit asyncio issue [temporary fix]
# https://github.com/streamlit/streamlit/issues/744#issuecomment-686712930
from duckduckgo_search import DDGS
from nltk.corpus import stopwords
lemmatizer = WordNetLemmatizer()

configs = config.load_config()

# ENHANCED_RETRIEVER with SEMANTIC-SEARCH & TF-IDF KeyWord based fine-sorting
max_return_docs = configs['retrieval']['max_return_docs']
initial_retrieval_count = configs['retrieval']['initial_retrieval_count']
stop_words = set(stopwords.words('english'))
ENABLE_HyDE = configs['retrieval']['enable_hyde']

# ColBERT-v2 Re-Ranker using RAGatouille
# ragatouille [& hence colbert-reranker] is not supported in windows
system = platform.system()
if not system == "Windows":
    from ragatouille import RAGPretrainedModel
    RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0") 
    RAG.model.inference_ckpt_len_set=False
    RAG.model.config.query_maxlen=600
    RAG.model.config.doc_maxlen=8192
    RAG.model.config.overwrite=True


def preprocess_text(document):
    # Extract text content from the Document
    # Convert to lowercase
    content = document.lower()
    # Tokenize and remove stop words
    tokens = [word for word in content.split() if word not in stop_words]
    # Lemmatize
    lemmatized_text = ' '.join(
        [lemmatizer.lemmatize(token) for token in tokens])
    return lemmatized_text

# DOC REMOVE duplicates from langchain documents

import re
from langchain.docstore.document import Document

AFFIL_PATTERN = re.compile(
    r"\\(author|affiliation|documentclass|usepackage|orcidlink)", re.I
)

def looks_like_noise(text: str) -> bool:
    # lots of back-slashes usually means LaTeX boiler-plate
    slash_density = text.count("\\") / max(len(text), 1)
    return slash_density > 0.02 or bool(AFFIL_PATTERN.search(text))

# filter out latex affiliation, institute etc
def filter_latex_boiler_plate(docs_with_scores: list[tuple]) -> list[tuple]:
    return [
        (doc, score)
        for doc, score in docs_with_scores
        if not looks_like_noise(doc.page_content)
    ]

def remove_duplicate_docs(my_list):
    unique_list = []
    for item in my_list:
        if item not in unique_list:
            unique_list.append(item)
    return unique_list

# Define a function to check if a query term matches any term in the document content


def is_fuzzy_match(query_term, doc_content):
    # Use fuzz.partial_ratio for partial matching (e.g., "titania" and "titanium-germanium")
    # You can adjust the threshold as needed (e.g., fuzz.partial_ratio(query_term, doc_term) >= 80)
    threshold = 80
    return any(fuzz.partial_ratio(query_term, doc_term) >= threshold for doc_term in doc_content.split())


def modify_question_using_HyDE(query):
    # HyDE: Augment Query with hypothetical answer
    # query = query + " . "+ st.session_state.myLLM(f'generate a concise single line answer (without any equations) that can be used to answer the ultimate technical question asked in this paragraph. {query}')
    query = query + " . " + st.session_state.myLLM(
        f'for the given question, generate a hypothetical concise single line answer (without any equations) as if you were a subject expert. Here is the question:  {query}')
    print("\n##################\n")
    print("Modified HyDE Query \n")
    print(query)
    print("##################\n")


# arxiv data fetch


def fetch_arxiv_abstracts(keyword="LIGO", start_date_str="2014-01-01", end_date_str="2024-01-31", max_results_per_year=10000, max_try=3):
    #############################
    # EXAMPLE:
    #############################
    # Define the date range (last two years from now)
    # end_date = datetime.now()
    # start_date = end_date - timedelta(days=2*365)  # Adjust for leap years if necessary
    # Fetch abstracts with keyword LIGO in the date range
    # keyword = "LIGO"
    # arXiv_docs = fetch_arxiv_abstracts("LIGO", start_date, end_date, max_results=10000)
    ##
    # Save abstracts and dates to a file
    # with open("arxiv_abstracts_with_dates.txt", "w") as file:
    # for title, abstract, published, updated in articles:
    # file.write(f"Title: {title}\nPublished: {published}\nUpdated: {updated}\nAbstract: {abstract}\n\n")
    # print(f"Saved {len(articles)} articles to arxiv_abstracts_with_dates.txt")
    ##
    #############################
    # Base URL for arXiv API
    base_arxiv_api_url = "http://export.arxiv.org/api/query?"
    # Formatting dates for the query
    # Convert strings to datetime objects
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    def fetch_articles(segment_start_str, segment_end_str):
        # Parameters for the query
        query = f"all:{keyword} AND submittedDate:[{segment_start_str} TO {segment_end_str}]"
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results_per_year
        }
        # Send a GET request to the arXiv API
        response = requests.get(base_arxiv_api_url, params=params)
        if response.status_code != 200:
            print("Error fetching data from arXiv")
            return
        # Parse the XML response
        root = ET.fromstring(response.content)
        # Extract and print abstracts along with dates
        articles = []
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title = entry.find('{http://www.w3.org/2005/Atom}title').text
            abstract = entry.find('{http://www.w3.org/2005/Atom}summary').text
            published_date = entry.find(
                '{http://www.w3.org/2005/Atom}published').text
            updated_date = entry.find(
                '{http://www.w3.org/2005/Atom}updated').text
            id_link = entry.find('{http://www.w3.org/2005/Atom}id').text
            arxiv_id = id_link.split('/abs/')[-1]
            # Extract version-independent arXiv ID
            doi = f"https://doi.org/10.48550/arXiv.{arxiv_id.split('v')[0]}"
            articles.append(
                (title, abstract, published_date, updated_date, doi))
        return articles
    # split to yearwise
    # Iterate over each year
    current_date = start_date
    articles = []
    documents = []
    while current_date.year <= end_date.year:
        # Determine the start and end of the current year segment
        segment_start = current_date
        if current_date.year == end_date.year:
            segment_end = end_date
        else:
            segment_end = datetime(current_date.year, 12, 31)
        segment_start_str = segment_start.strftime('%Y%m%d')
        segment_end_str = segment_end.strftime('%Y%m%d')
        count = 1
        articles = []
        # get best of ten trials, because of network uncertaininty
        print(
            f"Fetching ArXiv article abstract info btw {segment_start_str} - {segment_end_str}")
        while count < max_try:
            temp_articles = fetch_articles(segment_start_str, segment_end_str)
            if count == 1:
                articles = temp_articles
            else:
                if len(temp_articles) > len(articles):
                    articles = temp_articles
            count = count+1
        print(f"Obtained {len(articles)} articles")
        # Move to the next year
        current_date = datetime(current_date.year+1, current_date.month, 1)
        for title, abstract, published, updated, doi in articles:
            # Combine information into page content
            page_content = f"Title: {title}\nPublished: {published}\nUpdated: {updated}\nDOI: {doi}\nAbstract:{abstract}"
            # Create a LangChain Document
            document = Document(page_content=page_content, metadata={
                                'title': title, 'published': published, 'updated': updated, 'doi': doi})
            documents.append(document)
    print(
        f"Final number of retreived documents btw {start_date_str} - {end_date_str}: {len(documents)}")
    return documents


# ENHANCED_RETRIEVER with SEMANTIC-SEARCH & TF-IDF KeyWord based fine-sorting
def enhanced_retriever_ini(query, vectorstore, max_return_docs=max_return_docs, initial_retrieval_count=initial_retrieval_count, search_type="mmr"):
    # Initial semantic retrieval
    retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs={
                                         "k": initial_retrieval_count})
    semantic_docs = retriever.get_relevant_documents(query)
    # Extract texts for TF-IDF
    doc_texts = [doc.page_content for doc in semantic_docs]
    # TF-IDF processing
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(doc_texts)
    query_vector = tfidf_vectorizer.transform([query])
    cosine_similarities = cosine_similarity(
        query_vector, tfidf_matrix).flatten()
    # Sort and select top documents
    sorted_doc_indices = cosine_similarities.argsort()[::-1]
    top_docs = [Document(page_content=semantic_docs[i].page_content, metadata={
                         "source": "semantic_search"}) for i in sorted_doc_indices[:max_return_docs]]
    return top_docs


# Semantic+TF-IDF+Lexical+KeyWord-Filter
def enhanced_retriever(query, vectorstore, max_return_docs=max_return_docs, initial_retrieval_count=initial_retrieval_count, search_type="mmr"):
    # preprocess query
    query = preprocess_text(query)
    # Initial semantic retrieval
    retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs={
                                         "k": initial_retrieval_count})
    semantic_docs = retriever.get_relevant_documents(query)
    # remove duplicate documents
    # TO FIX: WHY SO MANY DUPLICATES
    semantic_docs = remove_duplicate_docs(semantic_docs)
    # REMOVE the last DOC with weird message "'pulling it low. The unit requires"
    # TO FIX: WHY is this happening
    semantic_docs = semantic_docs[1:-1]
    # Extract texts for TF-IDF
    doc_texts = [doc.page_content for doc in semantic_docs]
    if configs['retrieval']['enable_verbose']:
        print("\n#############  OLLAMA EMBEDDING based DOC SIMILARITY ###############")
        print("\n########################################")
        print(doc_texts)
        print("\n########################################")
    # TF-IDF processing
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(doc_texts)
    query_vector = tfidf_vectorizer.transform([query])
    cosine_similarities = cosine_similarity(
        query_vector, tfidf_matrix).flatten()
    # Sort documents based on TF-IDF cosine similarity
    tfidf_sorted_doc_indices = cosine_similarities.argsort()[::-1]
    if configs['retrieval']['enable_verbose']:
        print("\n#############  OLLAMA EMBEDDING + TF-IDF FineSort ###############")
        print("\n########################################")
        print([semantic_docs[idx].page_content for idx in tfidf_sorted_doc_indices])
        print("\n########################################")
    # Apply fuzzy matching and filter documents
    fuzzy_scores = [fuzz.token_sort_ratio(
        query, semantic_docs[idx].page_content) for idx in tfidf_sorted_doc_indices]
    fuzzy_sorted_doc_indices = [idx for _, idx in sorted(zip(
        fuzzy_scores, tfidf_sorted_doc_indices), key=lambda pair: pair[0], reverse=True)]
    filtered_docs = [(semantic_docs[idx].page_content, set())
                     for idx in fuzzy_sorted_doc_indices[:max_return_docs]]
    if configs['retrieval']['enable_verbose']:
        print("\n#############  OLLAMA EMBEDDING + TF-IDF FineSort + FuzzySort  ###############")
        print("\n########################################")
        print([(semantic_docs[idx].page_content, set())
              for idx in fuzzy_sorted_doc_indices[:max_return_docs]])
        print("\n########################################")
    # Extract key terms from the query (simple split, can be more complex)
    if configs['retrieval']['keyword_search_type'] == "fuzzy":
        query_terms = set(query.lower().split())
        # Filter documents based on the presence of query terms and fuzzy matching
        filtered_docs = []
        for idx in fuzzy_sorted_doc_indices:
            doc_content = semantic_docs[idx].page_content.lower()
            # preprocess doc_content
            # doc_content = preprocess_text(doc_content)
            doc_terms = set(
                [word for word in doc_content.split() if word not in stop_words])
            # Check for term overlap using fuzzy matching
            fuzzy_overlap = [
                term for term in query_terms if is_fuzzy_match(term, doc_content)]
            if fuzzy_overlap:
                # Highlight overlapping terms in the document content
                highlighted_content = semantic_docs[idx].page_content
                for term in fuzzy_overlap:
                    highlighted_content = highlighted_content.replace(
                        term, f"{term}")
                filtered_docs.append((highlighted_content, fuzzy_overlap))
            if len(filtered_docs) >= max_return_docs:
                break
        # Check if filtered_docs is empty
        if not filtered_docs:
            print("Semantic search Warning (ollma embbdeding based): No documents found after keyword-based filtering. Returning top documents from fuzzy matching.")
            filtered_docs = [(semantic_docs[idx].page_content, set())
                             for idx in fuzzy_sorted_doc_indices]
        else:
            # Print results with highlighted overlap
            print("\n############# OLLAMA EMBEDDING + TF-IDF FineSort + FuzzySort + KeyWord Filtering ###############")
            print("\n########################################")
            for doc, overlap in filtered_docs:
                print(f"Document with overlap {overlap}:")
                print(doc)
                print("\n")
        # Return the top filtered documents without highlighting
        top_docs = [Document(page_content=doc, metadata={
                             "source": "semantic_search"}) for doc, _ in filtered_docs]
        return top_docs
    elif configs['retrieval']['keyword_search_type'] == "exact":
        query_terms = set(query.lower().split())
        # Filter documents based on the presence of query terms
        filtered_docs = []
        for idx in fuzzy_sorted_doc_indices:
            doc_content = semantic_docs[idx].page_content.lower()
            # preprocess doc_content
            # doc_content = preprocess_text(doc_content)
            doc_terms = set(
                [word for word in doc_content.split() if word not in stop_words])
            overlap = query_terms.intersection(doc_terms)
            if overlap:  # Check for term overlap
                # Highlight overlapping terms in the document content
                highlighted_content = semantic_docs[idx].page_content
                for term in overlap:
                    highlighted_content = highlighted_content.replace(
                        term, f"**{term}**")
                filtered_docs.append((highlighted_content, overlap))
                if len(filtered_docs) >= max_return_docs:
                    break
        # Check if filtered_docs is empty
        if not filtered_docs:
            print("Semantic search Warning (ollma embbdeding based) : No documents found after keyword-based filtering. Returning top documents from fuzzy matching.")
            filtered_docs = [(semantic_docs[idx].page_content, set())
                             for idx in fuzzy_sorted_doc_indices[:max_return_docs]]
        else:
            # Print results with highlighted overlap
            print("\n#############  OLLAMA EMBEDDING + TF-IDF FineSort + FuzzySort + KeyWord Filtering ###############")
            print("\n########################################")
            for doc, overlap in filtered_docs:
                print(f"Document with overlap {overlap}:")
                print(doc)
                print("\n")
        # Return the top filtered documents without highlighting
        top_docs = [Document(page_content=doc, metadata={
                             "source": "semantic_search"}) for doc, _ in filtered_docs]
        return top_docs


def verify_and_filter_retrieved_docs(user_query, docs):
    filtered_retrieved_docs = []
    for doc in docs:
        # Extract the filename from metadata if present
        filename = doc.metadata.get('source_file', 'N/A')
        # Prepare the prompt with the actual filename
        prompt = (
            f"Look for the answer to the QUESTION solely based on this CONTEXT. "
            f"Mention the answer if found within the context along with the context-based rationale behind your answer. "
            f"Do not add any extra information not provided in the CONTEXT. "
            f"Return <NIL> info if CONTEXT is insufficient. "
            f"Include << FILENAME >> if present in metadata.\n"
            f"CONTEXT: {doc.page_content}\n"
            f"METADATA: {doc.metadata}\n"
            f"QUESTION: {user_query}\n"
            f"OUTPUT:"
        )
        # Get the LLM output
        llm_output = st.session_state.llm(prompt).strip()
        # Check if 'nil' is not in the LLM output (case-insensitive)
        if 'nil' not in llm_output.lower():
            # Create a new Document with the LLM output as page_content and original metadata
            new_doc = Document(page_content=llm_output, metadata=doc.metadata)
            filtered_retrieved_docs.append(new_doc)
    return filtered_retrieved_docs

def remove_duplicate_docs_span_aware(docs):
    seen = set()
    out = []
    for d in docs:
        key = (
            d.metadata.get("span_source"),
            d.metadata.get("span_start_idx"),
            d.metadata.get("span_end_idx"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def verify_and_filter_retrieved_docs_v2_parallel(user_query: str, docs: list[Document], useGROQ: bool = True, max_workers=10, use_threads: bool = True):
    """
    Given a user_query and a list of Document objects, this function checks
    whether each Document contains relevant information. If it does, we keep it;
    if not, we discard it. DO NOT expand acronyms or make guesses.

    Args:
        user_query (str): The query to evaluate document relevance against.
        docs (list[Document]): List of Document objects to process.
        useGROQ (bool): If True, process documents with Groq API; if False, process sequentially (default: True).
        max_workers (int): Number of worker threads for parallel processing (default: 10).
        use_threads (bool): If True, use threads for parallel processing; if False, use sequential loop (default: True).

    Returns:
        tuple: (filtered_retrieved_docs, filtered_retrieved_docs_LLM_answers)
    """
    import time  # For timing in verbose output

    # Retrieve LLM from session_state in the main thread
    #if 'llm' not in st.session_state:
    #    raise ValueError("LLM not found in st.session_state. Please initialize it first.")
    
    llm = st.session_state.llm

    # Few-shot examples for LLM prompting
    few_shot_examples_content = get_prompt("VERIFY_DOC_FEW_SHOTS", {})

    # Nested helper function to process a single document
    def _process_single_doc(doc, llm, idx=None, verbose=False):
        try:
            if verbose:
                print(f"Starting processing for document {idx}...")
                start_time = time.time()
            
            filename = doc.metadata.get('source_file') or doc.metadata.get('source') or "N/A"

            prompt_text = get_prompt(
                "VERIFY_DOC_RELEVANCE_template",
                {
                    "few_shot_examples": few_shot_examples_content,
                    "filename": filename,
                    "context_content": doc.page_content,
                    "user_query": user_query
                }
            )
            
            # Check Doc for relevancy
            llm_output = llm(prompt_text).strip()

            if "<NIL>" not in llm_output :
                
                if "<CHECK_URL>"  in llm_output  :
                    # Process Potentially Relevant Document by extracting URL Info        
                    url = str(extract_url_from_Tavily_Doc_source_metadata(doc))
                    print(f"Checking URL data: {url} from the retrieved doc, for more info.")    
                    if configs['retrieval']['ENABLE_WEB_PDF_EXTRACTION']:
                        url_data = get_text_from_url_including_pdf(url, max_tokens=configs['retrieval']['WEB_EXTRACTION_TOKEN_LIMIT']) or ""   # ← N is user-defined                        
                    else:                
                        url_data = str(extract_text_from_url(url))        
                    new_doc = Document(page_content=doc.page_content + "\n-------------\n" + url_data, metadata=doc.metadata)  
                else:  
                    # Process Already Relevant Doc
                    new_doc = Document(page_content=doc.page_content + "\n-------------\n" + llm_output, metadata=doc.metadata)

                
                prompt_text = get_prompt(
                    "REPHRASE_CHUNK_STANDALONE_template",
                    {
                        "user_query": user_query,
                        "text_chunk": str(new_doc) 
                    }
                )
                new_doc.page_content = llm(prompt_text)
                
                new_doc.page_content = wrap_urls_in_angle_brackets(new_doc.page_content)
                try:
                    new_doc.metadata['source'] = update_LatexData_downloaded_arxiv_articles_id_with_PDF_URL(new_doc.metadata['source'])
                except:
                    pass
                try:
                    new_doc.metadata['url'] = update_LatexData_downloaded_arxiv_articles_id_with_PDF_URL(new_doc.metadata['url'])
                except:
                    pass
                try:
                    new_doc.page_content = replace_plus_in_url_paths(new_doc.page_content)
                except:
                    pass
                try:
                    new_doc.metadata['source'] = replace_plus_in_url_paths(new_doc.metadata['source'])
                except:
                    pass
                try:
                    new_doc.metadata['url'] = replace_plus_in_url_paths(new_doc.metadata['url'])
                except:
                    pass
                try:
                    new_doc.metadata['source'] = wrap_urls_in_angle_brackets(new_doc.metadata['source'])
                except:
                    pass
                try:
                    new_doc.metadata['url'] = wrap_urls_in_angle_brackets(new_doc.metadata['url'])
                except:
                    pass
                try:
                    wrap_urls_in_metadata(new_doc)
                except:
                    pass
                try:
                    new_doc.page_content = hyperlink_dcc_urls(new_doc.page_content)
                except:
                    pass
                try:
                    new_doc.metadata['source'] = hyperlink_dcc_urls(new_doc.metadata['source'])
                except:
                    pass
                try:
                    new_doc.metadata['url'] = hyperlink_dcc_urls(new_doc.metadata['url'])
                except:
                    pass
                try:
                    new_doc.page_content = nicely_format_latex_eqns(new_doc.page_content)
                except:
                    pass                

                if verbose:
                    end_time = time.time()
                    duration = end_time - start_time
                    print(f"Finished processing for document {idx}. Time taken: {duration:.2f} seconds.")
                return doc, new_doc
                
            else:
                if verbose:
                    end_time = time.time()
                    duration = end_time - start_time
                    print(f"Document {idx} is not relevant. Time taken: {duration:.2f} seconds.")
                return None
        except Exception as e:
            if verbose:
                end_time = time.time()
                duration = end_time - start_time
                print(f"Error processing document {idx}: {e}. Time taken: {duration:.2f} seconds.")
            else:
                print(f"Error processing document: {e}")
            return None

    filtered_retrieved_docs = []
    filtered_retrieved_docs_LLM_answers = []

    if useGROQ:
        print(f"Starting processing with Groq API...")
        if use_threads:
            from concurrent.futures import ThreadPoolExecutor
            print(f"Using parallel threading with {max_workers} workers...")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                tasks = [(doc, idx) for idx, doc in enumerate(docs)]
                results = list(executor.map(lambda task: _process_single_doc(task[0], llm, task[1], verbose=True), tasks))
            print("Parallel threading processing completed.")
        else:
            print("Using sequential Groq API calls with verbose output...")
            results = []
            for idx, doc in enumerate(docs):
                result = _process_single_doc(doc, llm, idx, verbose=True)
                results.append(result)
            print("Sequential Groq API processing completed.")
    else:
        results = [_process_single_doc(doc, llm) for doc in docs]

    for result in results:
        if result is not None:
            original_doc, processed_doc = result
            filtered_retrieved_docs.append(original_doc)
            filtered_retrieved_docs_LLM_answers.append(processed_doc)
            #streamlit_add_line(st=st)
            #streamlit_add_msg(st=st, role="assistant-0", message=f"Document {st.session_state.doc_counter}\n")
            #streamlit_add_msg(st=st, role="assistant-0", message=processed_doc)
            st.session_state.doc_counter += 1

    return filtered_retrieved_docs, filtered_retrieved_docs_LLM_answers



def verify_and_filter_retrieved_docs_v2_parallel_api_version(user_query: str, docs: list[Document], useGROQ: bool = True, max_workers=10, use_threads: bool = True,st=None):
    """
    Given a user_query and a list of Document objects, this function checks
    whether each Document contains relevant information. If it does, we keep it;
    if not, we discard it. DO NOT expand acronyms or make guesses.

    Args:
        user_query (str): The query to evaluate document relevance against.
        docs (list[Document]): List of Document objects to process.
        useGROQ (bool): If True, process documents with Groq API; if False, process sequentially (default: True).
        max_workers (int): Number of worker threads for parallel processing (default: 10).
        use_threads (bool): If True, use threads for parallel processing; if False, use sequential loop (default: True).

    Returns:
        tuple: (filtered_retrieved_docs, filtered_retrieved_docs_LLM_answers)
    """
    import time  # For timing in verbose output

    # Retrieve LLM from session_state in the main thread
    #if 'llm' not in st.session_state:
    #    raise ValueError("LLM not found in st.session_state. Please initialize it first.")
    
    llm = st.session_state.llm

    # Few-shot examples for LLM prompting
    few_shot_examples_content = get_prompt("VERIFY_DOC_FEW_SHOTS", {})

    # Nested helper function to process a single document
    def _process_single_doc(doc, llm, idx=None, verbose=False):
        try:
            if verbose:
                print(f"Starting processing for document {idx}...")
                start_time = time.time()
            
            filename = doc.metadata.get('source_file') or doc.metadata.get('source') or "N/A"
            
            prompt_text = get_prompt(
                "VERIFY_DOC_RELEVANCE_template",
                {
                    "few_shot_examples": few_shot_examples_content,
                    "filename": filename,
                    "context_content": doc.page_content,
                    "user_query": user_query
                }
            )

            # Check Doc for relevancy
            llm_output = llm(prompt_text).strip()
            
            if "<NIL>" not in llm_output :
                
                if "<CHECK_URL>"  in llm_output  :
                    # Process Potentially Relevant Document by extracting URL Info        
                    url = str(extract_url_from_Tavily_Doc_source_metadata(doc))
                    print(f"Checking URL data: {url} from the retrieved doc, for more info.")    
                    if configs['retrieval']['ENABLE_WEB_PDF_EXTRACTION']:
                        url_data = get_text_from_url_including_pdf(url, max_tokens=configs['retrieval']['WEB_EXTRACTION_TOKEN_LIMIT']) or ""   # ← N is user-defined                        
                    else:                
                        url_data = str(extract_text_from_url(url))        
                    new_doc = Document(page_content=doc.page_content + "\n-------------\n" + url_data, metadata=doc.metadata)  
                else:  
                    # Process Already Relevant Doc
                    new_doc = Document(page_content=doc.page_content + "\n-------------\n" + llm_output, metadata=doc.metadata)

                prompt_text = get_prompt(
                    "REPHRASE_CHUNK_STANDALONE_template",
                    {
                        "user_query": user_query,
                        "text_chunk": str(new_doc) 
                    }
                )
                new_doc.page_content = llm(prompt_text)
                
                new_doc.page_content = wrap_urls_in_angle_brackets(new_doc.page_content)
                try:
                    new_doc.metadata['source'] = update_LatexData_downloaded_arxiv_articles_id_with_PDF_URL(new_doc.metadata['source'])
                except:
                    pass
                try:
                    new_doc.metadata['url'] = update_LatexData_downloaded_arxiv_articles_id_with_PDF_URL(new_doc.metadata['url'])
                except:
                    pass
                try:
                    new_doc.page_content = replace_plus_in_url_paths(new_doc.page_content)
                except:
                    pass
                try:
                    new_doc.metadata['source'] = replace_plus_in_url_paths(new_doc.metadata['source'])
                except:
                    pass
                try:
                    new_doc.metadata['url'] = replace_plus_in_url_paths(new_doc.metadata['url'])
                except:
                    pass
                try:
                    new_doc.metadata['source'] = wrap_urls_in_angle_brackets(new_doc.metadata['source'])
                except:
                    pass
                try:
                    new_doc.metadata['url'] = wrap_urls_in_angle_brackets(new_doc.metadata['url'])
                except:
                    pass
                try:
                    wrap_urls_in_metadata(new_doc)
                except:
                    pass
                try:
                    new_doc.page_content = hyperlink_dcc_urls(new_doc.page_content)
                except:
                    pass
                try:
                    new_doc.metadata['source'] = hyperlink_dcc_urls(new_doc.metadata['source'])
                except:
                    pass
                try:
                    new_doc.metadata['url'] = hyperlink_dcc_urls(new_doc.metadata['url'])
                except:
                    pass
                try:
                    new_doc.page_content = nicely_format_latex_eqns(new_doc.page_content)
                except:
                    pass                

                if verbose:
                    end_time = time.time()
                    duration = end_time - start_time
                    print(f"Finished processing for document {idx}. Time taken: {duration:.2f} seconds.")
                return doc, new_doc
                
            else:
                if verbose:
                    end_time = time.time()
                    duration = end_time - start_time
                    print(f"Document {idx} is not relevant. Time taken: {duration:.2f} seconds.")
                return None
        except Exception as e:
            if verbose:
                end_time = time.time()
                duration = end_time - start_time
                print(f"Error processing document {idx}: {e}. Time taken: {duration:.2f} seconds.")
            else:
                print(f"Error processing document: {e}")
            return None

    filtered_retrieved_docs = []
    filtered_retrieved_docs_LLM_answers = []

    if useGROQ:
        print(f"Starting processing with Groq API...")
        if use_threads:
            from concurrent.futures import ThreadPoolExecutor
            print(f"Using parallel threading with {max_workers} workers...")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                tasks = [(doc, idx) for idx, doc in enumerate(docs)]
                results = list(executor.map(lambda task: _process_single_doc(task[0], llm, task[1], verbose=True), tasks))
            print("Parallel threading processing completed.")
        else:
            print("Using sequential Groq API calls with verbose output...")
            results = []
            for idx, doc in enumerate(docs):
                result = _process_single_doc(doc, llm, idx, verbose=True)
                results.append(result)
            print("Sequential Groq API processing completed.")
    else:
        results = [_process_single_doc(doc, llm) for doc in docs]

    for result in results:
        if result is not None:
            original_doc, processed_doc = result
            filtered_retrieved_docs.append(original_doc)
            filtered_retrieved_docs_LLM_answers.append(processed_doc)
            #streamlit_add_line(st=st)
            #streamlit_add_msg(st=st, role="assistant-0", message=f"Document {st.session_state.doc_counter}\n")
            #streamlit_add_msg(st=st, role="assistant-0", message=processed_doc)
            st.session_state.doc_counter += 1

    return filtered_retrieved_docs, filtered_retrieved_docs_LLM_answers

def verify_and_filter_retrieved_docs_v2(user_query: str, docs: list[Document]):
    """
    Given a user_query and a list of Document objects, this function checks
    whether each Document contains relevant information. If it does, we keep it;
    if not, we discard it. DO NOT expand acronyms or make guesses. 
    For exaample, is query is about ISI, dont expand the term to say Initial Titanium Isolator blah blab

    The logic is delegated to an LLM, which must return a structured output:
    - ANALYSIS: <some text or <NIL>>
    - RATIONALE: <short rationale>

    If the ANALYSIS is <NIL>, we discard the Document. Otherwise, we keep it,
    storing the entire LLM response as the new Document's page_content.
    """

    # A few-shot prompt section with explicit examples
    simple_few_shots = get_prompt("VERIFY_DOC_SIMPLE_FEW_SHOTS", {})

    filtered_retrieved_docs = []
    filtered_retrieved_docs_LLM_answers = []    

    for doc in docs:
        # Pull the filename from metadata if present
        filename = (
            doc.metadata.get('source_file') 
            or doc.metadata.get('source') 
            or "N/A"
        )

        prompt_text = get_prompt(
            "VERIFY_DOC_SIMPLE_RELEVANCE_template",
            {
                "filename": filename,
                "few_shot_examples": simple_few_shots,
                "context_content": doc.page_content,
                "user_query": user_query
            }
        )

        # Call your LLM
        print_wide_line_message(doc)
        if st.session_state.useGroq:
            llm_output = st.session_state.llm(prompt_text).strip()
        else:
            llm_output = st.session_state.llm(prompt_text).strip()
        print("\n---------------------------------------\n")

        # If <NIL> appears anywhere in the LLM output, we discard this doc.
        # Otherwise, we keep the doc.
        if "<NIL>" not in llm_output:
            new_doc = Document(page_content=doc.page_content, metadata=doc.metadata)
            filtered_retrieved_docs.append(new_doc)

            new_doc = Document(page_content=doc.page_content+"\n-------------\n"+llm_output, metadata=doc.metadata)


            # Post-Processing new_doc using LLM
            if st.session_state.useGroq:
                new_doc.page_content =     st.session_state.llm(f'''
                                reprase this text chunk into a standalone single paragraph such that it retains 
                                all the information useful in answering the query. Donot try to answer the query.
                                Include all the technical details including numbers, source info, equations etc 
                                present in the original text chunk. Donot repeat information. Donot expand any acronyms.
                                Always show the Metadata Source  (including file extension) within <>.   
                                Enclose short LaTex equations within $...$ (Single Dollar Signs) for inline math mode and more complex LaTex equations within $$...$$ (Double Dollar Signs) for block math mode.
                                Here is the query {user_query}. \n 
                                Here is the text chunk: {new_doc}
                                ''')
            else:
                new_doc.page_content =     st.session_state.llm(f'''
                                reprase this text chunk into a standalone single paragraph such that it retains 
                                all the information useful in answering the query. Donot try to answer the query.
                                Include all the technical details including numbers, source info, equations etc 
                                present in the original text chunk. Donot repeat information. Donot expand any acronyms.
                                Always show the Metadata Source  (including file extension) within <>.   
                                Enclose short LaTex equations within $...$ (Single Dollar Signs) for inline math mode and more complex LaTex equations within $$...$$ (Double Dollar Signs) for block math mode.                                                                
                                Here is the query {user_query}. \n 
                                Here is the text chunk: {new_doc}
                                ''')                
            # make sure urls are within <> tags
            #breakpoint()
            new_doc.page_content = wrap_urls_in_angle_brackets(new_doc.page_content)

            # METADATA URL PostProcessing
            # update_LatexData_downloaded_arxiv_articles_id_with_PDF_URL
            ##---------------------------------------------------------------------------##
            try:
                new_doc.metadata['source'] = update_LatexData_downloaded_arxiv_articles_id_with_PDF_URL(new_doc.metadata['source'])
            except:
                pass
            ##---------------------------------------------------------------------------##
            try:
                new_doc.metadata['url'] = update_LatexData_downloaded_arxiv_articles_id_with_PDF_URL(new_doc.metadata['url'])
            except:
                pass       
            ##---------------------------------------------------------------------------##
            #  replace_plus_in_url_paths
            try:
                new_doc.page_content = replace_plus_in_url_paths(new_doc.page_content)
            except:
                pass                  
            ##---------------------------------------------------------------------------##
            #  replace_plus_in_url_paths
            try:
                new_doc.metadata['source'] = replace_plus_in_url_paths(new_doc.metadata['source'])
            except:
                pass
            ##---------------------------------------------------------------------------##
            try:
                new_doc.metadata['url'] = replace_plus_in_url_paths(new_doc.metadata['url'])
            except:
                pass            
            ##---------------------------------------------------------------------------##
            # wrap_urls_in_angle_brackets
            try:
                new_doc.metadata['source'] = wrap_urls_in_angle_brackets(new_doc.metadata['source'])
            except:
                pass
            ##---------------------------------------------------------------------------##
            try:
                new_doc.metadata['url'] = wrap_urls_in_angle_brackets(new_doc.metadata['url'])
            except:
                pass                
            ##---------------------------------------------------------------------------##
            # Ensure 'URLs' are correctlty '<URL>' formatted
            try:
                wrap_urls_in_metadata(new_doc)    
            except:
                pass
            ##---------------------------------------------------------------------------##
            ## Update & hyperlink DCC urls
            ##---------------------------------------------------------------------------##
            try:
                new_doc.page_content = hyperlink_dcc_urls(new_doc.page_content)
            except:
                pass
            try:
                new_doc.metadata['source'] = hyperlink_dcc_urls(new_doc.metadata['source'])
            except:
                pass
            ##---------------------------------------------------------------------------##
            try:
                new_doc.metadata['url'] = hyperlink_dcc_urls(new_doc.metadata['url'])
            except:
                pass            
            ##---------------------------------------------------------------------------##            
            try:
                new_doc.page_content = nicely_format_latex_eqns(new_doc.page_content)
            except:
                pass                 


            # Show Doc in Streamlit output
            #streamlit_add_line(st=st)
            #streamlit_add_msg(st=st,role="assistant-0",message=f"Document {st.session_state.doc_counter}\n")
            #streamlit_add_msg(st=st,role="assistant-0",message=new_doc)    



            # Append filtered_retrieved_docs_LLM_answers
            filtered_retrieved_docs_LLM_answers.append(new_doc)

            # Append st.session_state.doc_counter
            st.session_state.doc_counter = st.session_state.doc_counter + 1
            

    return filtered_retrieved_docs,filtered_retrieved_docs_LLM_answers



# Semantic (Ollam-ligoGPT embedding) + KeyWord(TF-IDF) search


class CustomRetriever(VectorStoreRetriever):
    search_type: str = "mmr"
    search_kwargs: dict = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True  # Allow custom types

    def __init__(self, vectorstore, search_type, search_kwargs, **data):
        super().__init__(vectorstore=vectorstore, retriever=None, **data)
        self.vectorstore = vectorstore
        self.search_type = search_type
        self.search_kwargs = search_kwargs

    def get_relevant_documents(self, query: str) -> List[Document]:
        top_docs = enhanced_retriever(query, self.vectorstore, max_return_docs=max_return_docs,
                                      initial_retrieval_count=initial_retrieval_count, search_type="mmr")
        return top_docs


def enhanced_retriever_BM25(query, vectorstore, max_return_docs=max_return_docs, initial_retrieval_count=initial_retrieval_count, n_terms=configs['retrieval']['BM25_min_terms']):
    # Function to check if a document contains at least two query terms
    def contains_at_least_N_terms(document, user_input_terms, n_terms=n_terms):
        term_count = sum(term.lower() in word_tokenize(
            (document.page_content).lower()) for term in user_input_terms)
        if configs['retrieval']['enable_verbose']:
            if term_count >= n_terms:
                print("################################################")
                print(
                    f"####### BM25 Alert! Doc has alteast {n_terms} overlapping terms with the user_input ##########")
                print("################################################")
                print(document)
        return term_count >= n_terms

    def sanitizeText(input_string):
        # chars_to_replace = ['-', '!', '#', ".", ",", "(", ")", "{", "}", "[", "]", "&", "$", "%", "*", "@", "^", "+", "=", "/", "|", ":", ";", "<", ">", "?"]
        chars_to_replace = ['!', "<", ">", "?", ":"]
        regex_pattern = "[" + re.escape("".join(chars_to_replace)) + "]"
        output_string = re.sub(regex_pattern, " ", input_string)
        # old code
        # Replace all single characters (punctuation) with a space
        # output_string = re.sub(r'(?<=\s)[,.;!?](?=\s)|(?<=\s)[,.;!?]|[,.;!?](?=\s)', ' ', output_string)
        # convert to lower case
        output_string = output_string.lower()
        return output_string
    # Function to process each chunk





    def process_chunk(chunk, user_input_terms):
        bm25_retriever = BM25Retriever.from_documents(chunk)
        bm25_retriever.k = initial_retrieval_count
        # BM25
        retrieved_docs = bm25_retriever.get_relevant_documents(user_input)
        # remove duplicates
        retrieved_documents = remove_duplicate_docs(retrieved_docs)
        # post-filter via keyword filtering (atleast two)
        # Filter the documents
        if len(user_input_terms) > 1:
            # print(user_input_terms)
            filtered_documents = [doc for doc in retrieved_documents if contains_at_least_N_terms(
                doc, user_input_terms, n_terms=configs['retrieval']['BM25_min_terms'])]
        else:
            filtered_documents = [
                XYZ for XYZ in retrieved_documents if user_input_terms[0] in XYZ.page_content.lower()]
        return filtered_documents
    ##############
    user_input = query
    # breakpoint()
    # Define chunk size
    k = 5
    chunk_size = ceil(len(st.session_state.vectorstore_docs) / k)
    # clean user_uinput
    user_input = sanitizeText(user_input)
    # Split the user input into individual terms
    user_input_terms = word_tokenize(user_input.lower())
    # Your list of words to add
    additional_stop_words = ['desribe', 'tell', 'me', 'more', 'about', 'show', 'how', 'explain', 'ligo', 'gravitational', 'waves', 'gw', 'simple',
                             'expand', 'ok', 'perfect', 'great', 'cool', 'awesome', 'entry', 'logs', 'logbook', 'basic', 'info', 'scientists', 'use',
                             'dcc', 'fine', "'s", 'role', 'contribution', 'work', 'lab', 'search', 'docs', 'url', 'keyword', 'description', 'using', 'update',
                             'title', 'author', 'table', 'tabular', 'form', 'entries', 'KnowledgeBase', 'logbooks', 'find', 'phone', 'number', 'email', 'code']
    # Update the stop words set with your additional words
    stop_words.update(additional_stop_words)
    user_input_terms = [
        word for word in user_input_terms if word not in stop_words]
    if configs['retrieval']['enable_verbose']:
        print(f"Processed user_input for BM25:{user_input_terms}")
    # Splitting vectorstore_docs into chunks and process each chunk
    chunks = [st.session_state.vectorstore_docs[i:i + chunk_size]
              for i in range(0, len(st.session_state.vectorstore_docs), chunk_size)]
    combined_results = []
    count = 1
    for chunk in chunks:
        # print(f"processing chunk {count} of {len(chunks)} ")
        results = process_chunk(chunk, user_input_terms)
        combined_results.extend(results)
        count = count+1
    # Remove duplicates from the combined results
    top_docs = []
    try:
        top_docs = remove_duplicate_docs(combined_results)
        if len(top_docs) >= max_return_docs:
            top_docs = top_docs[:max_return_docs]
    except Exception as e:
        print("An error occurred during document retrieval and processing:")
        print(str(e))
        # Return an empty list if there's an error
        return []
    return top_docs

 # modified BM25 retriever


class CustomRetriever_BM25(VectorStoreRetriever):
    search_kwargs: dict = Field(default_factory=dict)   

    class Config:
        arbitrary_types_allowed = True  # Allow custom types

    def __init__(self, vectorstore, search_kwargs, **data):
        super().__init__(vectorstore=vectorstore, retriever=None, **data)
        self.vectorstore = vectorstore
        self.search_kwargs = search_kwargs

    def get_relevant_documents(self, query: str) -> List[Document]:
        top_docs = enhanced_retriever_BM25(
            query, self.vectorstore, max_return_docs=max_return_docs, initial_retrieval_count=initial_retrieval_count)
        return top_docs

# [ALERT OLD CODE, has duplicates:, USE ColBERT-FAISS, See Below]
# FAISS Retreiver with top_scores



# ------------------------------------------------------------------ #
# Enhanced FAISS retriever
# ------------------------------------------------------------------ #

def enhanced_retriever_FAISS(
    query,
    faiss_vectorstore,
    initial_retrieval_count=10,
    max_return_docs=5,
    enable_follow_urls=False,
):
    """
    Return at most *max_return_docs* LangChain Documents enriched with the
    fetched text of any URLs found inside them.

    Pass *enable_follow_urls=False* if you ever want the vanilla behaviour.
    """

    _URL_RE = re.compile(r"https?://\S+")


    def _extract_urls(text):
        """Return a list of all http(s) URLs found in *text*."""
        return _URL_RE.findall(text or "")


    def _fetch_page_text(url, timeout=10):
        """
        Fetch *url* and return its visible text.
        Falls back to an empty string if anything goes wrong.
        """
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            resp = requests.get(url, timeout=timeout, headers=headers)
            resp.raise_for_status()
        except Exception as exc:
            print(f"[warn] Could not retrieve {url!r}: {exc}")
            return ""

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove scripts / styles
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        # Collapse the remaining text
        return soup.get_text(separator=" ", strip=True)



    # 1️⃣  Vector search
    docs_and_scores = faiss_vectorstore.similarity_search_with_score(
        query, k=max_return_docs, fetch_k=initial_retrieval_count
    )

    # 2️⃣  Any post-filters you already had
    docs_and_scores = filter_latex_boiler_plate(docs_and_scores)

    docs = [doc for doc, score in docs_and_scores]
    docs = remove_duplicate_docs(docs)[:max_return_docs]
    
    # 3️⃣  OPTIONAL: follow any URLs and append their text
    if enable_follow_urls:
        for doc in docs:
            # Gather possible sources:
            urls = []
            # • from metadata
            if isinstance(doc.metadata, dict) and "source" in doc.metadata:
                urls.append(str(doc.metadata["source"]))
            # • from the chunk itself
            urls.extend(_extract_urls(doc.page_content))

            # De-duplicate while preserving order
            seen = set()
            for raw_url in urls:
                if raw_url in seen:
                    continue
                seen.add(raw_url)

                # Normalise any LIGO codes
                url = detect_and_convert_public_urls(raw_url, domain_keywords=None)

                fetched_txt = _fetch_page_text(url)

                if fetched_txt:
                    doc.page_content += (
                        f"\n\n---\n[Fetched from {url}]\n\n{fetched_txt}"
                    )

    # 4️⃣  Verbose debug print (unchanged)
    if configs["retrieval"]["enable_verbose"]:
        print("\nFAISS top docs (after enrichment)\n")
        for d in docs:
            print(d)

    return docs



def enhanced_retriever_FAISS_old(query, faiss_vectorstore, initial_retrieval_count=initial_retrieval_count, max_return_docs=max_return_docs):
    print("Executin enhanced_retriever_FAISS ")
    docs_and_scores = faiss_vectorstore.similarity_search_with_score(
        query, k=max_return_docs, fetch_k=initial_retrieval_count)
    # filter out latex affiliation, institute etc
    docs_and_scores = filter_latex_boiler_plate(docs_and_scores)
    # Splitting into separate lists
    top_docs = [doc for doc, score in docs_and_scores]
    top_docs = remove_duplicate_docs(top_docs)
    top_docs = top_docs[0:max_return_docs]
    if configs['retrieval']['enable_verbose']:
        print("\nFaiss top docs\n")
        print(top_docs)
    # top_docs_scores = [score for doc, score in docs_and_scores]
    return top_docs

# FAISS Retreiver with ColBERT Re-Ranking


def enhanced_retriever_FAISS_COlBERT(query, faiss_vectorstore, initial_retrieval_count=initial_retrieval_count, max_return_docs=max_return_docs):
    retriever = faiss_vectorstore.as_retriever(
        search_kwargs={"k": initial_retrieval_count})
    compression_retriever = ContextualCompressionRetriever(base_compressor=RAG.as_langchain_document_compressor(k=initial_retrieval_count),
                                                           base_retriever=retriever)
    compressed_docs = []
    compressed_docs = compression_retriever.get_relevant_documents(query)
    compressed_docs = remove_duplicate_docs(compressed_docs)
    top_docs = compressed_docs[0:max_return_docs]
    if configs['retrieval']['enable_verbose']:
        print("\nFaiss top docs\n")
        print(top_docs)
    return top_docs

# modified Custom FAISS retriever


class CustomRetriever_FAISS(VectorStoreRetriever):
    search_kwargs: dict = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True  # Allow custom types

    def __init__(self, vectorstore, search_kwargs, **data):
        super().__init__(vectorstore=vectorstore, retriever=None, **data)
        self.vectorstore = vectorstore
        self.search_kwargs = search_kwargs

    def get_relevant_documents(self, query: str) -> List[Document]:
        # HyDE: Augment Query with hypothetical answer
        if ENABLE_HyDE:
            query_HYDE = modify_question_using_HyDE(query)
        else:
            query_HYDE = query
        try:
            #top_docs = enhanced_retriever_FAISS(
            #    query_HYDE, self.vectorstore, initial_retrieval_count=initial_retrieval_count, max_return_docs=max_return_docs)

            top_docs = enhanced_retriever_FAISS(
                query_HYDE,
                faiss_vectorstore=self.vectorstore,
                initial_retrieval_count=initial_retrieval_count,
                max_return_docs=max_return_docs,
                enable_follow_urls=True,
                neighbor_span=0,  # <- e.g., [-2, +2]
                all_docs=st.session_state.vectorstore_docs,  # you already set this earlier
)            
        except:
            top_docs = enhanced_retriever_FAISS(
                query_HYDE, self.vectorstore, initial_retrieval_count=initial_retrieval_count, max_return_docs=max_return_docs)
        return top_docs


# Tavily WebSearch Retriever (requires API Key/ Free 1000 Requests per month)
def enhanced_retriever_tavily(query, max_return_docs=max_return_docs):
    if not os.environ.get("TAVILY_API_KEY"):
        os.environ["TAVILY_API_KEY"] = configs['retrieval']['TAVILY_API_KEY']    
    tool = TavilySearchResults(
        max_results=max_return_docs,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=True,
        include_images=True,
        # include_domains=[...],
        # exclude_domains=[...],
        # name="...",            # overwrite default tool name
        # description="...",     # overwrite default tool description
        # args_schema=...,       # overwrite default args_schema: BaseModel
    )    
    # Fetch results
    print_wide_line_message("Executing Tavily WebSearch")
    out = tool.invoke({"query": query})    
    # Convert to Langchain Document
    top_docs = []
    try:
        for idx, doc in enumerate(out):
            if isinstance(doc, dict):
                # Ensure 'url' and 'content' keys exist
                if 'url' in doc and 'content' in doc:
                    doc_url = detect_and_convert_public_urls(doc['url'])
                    page_content = f"{doc['content']}. URL/Source: {doc_url}"
                    metadata = {"source": doc_url}
                    
                    langchain_document = Document(
                        page_content=page_content, metadata=metadata
                    )                    
                    if configs['retrieval']['enable_verbose']:
                        print(f"Document {idx}: {langchain_document}")                    
                    top_docs.append(langchain_document)
                else:
                    print(f"Warning: Missing 'url' or 'content' in document {idx}: {doc}")
            else:
                print(f"Warning: Unexpected document format at index {idx}: {doc}")
    except Exception as e:
        print('#############################################################')
        print("Encountered issues while fetching data using Tavily WebSearch")
        traceback.print_exc()
        print('#############################################################')          
    return top_docs

# DuckDuckGo WebSearch Retriever
def enhanced_retriever_duckduckgo(query, max_return_docs=max_return_docs):
    # ADD LIGO & GW to the query to prevent unrelated content
    #
    # Query Compreession (DuckDuckGo doesnt like long queries)
    query_compressed = st.session_state.llm(
        f"Compress this query to a concise single line. Query:{query}")
    # remove double quptes frim query
    query_compressed = query_compressed.replace('"', '')

    print_wide_line_message(f"query_compressed for DuckdDuckGo via LLM: {query_compressed}")

    

    # for use with duckduckgo (max limit = 500 characters , 500/6(5charsper + 1 space) = 83)
    query_compressed = limit_text_to_first_N_words(query_compressed,N=80)
    top_docs = []
    try:
        with DDGS() as ddgs:
            duckduckgo_documents = [r for r in ddgs.text(
                query_compressed, max_results=max_return_docs)]
        top_docs = []
        for doc in duckduckgo_documents:
            title = str(doc.get('title', ''))
            href = str(doc.get('href', ''))
            body = str(doc.get('body', ''))
            page_content = f"Title: {title}\nURL: {href}\nBody: {body}"
            # Changed to 'source' as per your requirement
            metadata = {'source': href}
            langchain_document = Document(
                page_content=page_content, metadata=metadata)
            if configs['retrieval']['enable_verbose']:
                print(langchain_document)
            top_docs.append(langchain_document)
    except Exception:
        print('#############################################################')
        print("Encountered issues try to fetch data using DuckDuckGO")
        traceback.print_exc()
        print('#############################################################')
        pass
    return top_docs

 # modified Custom DuckDuckGo retriever

def limit_text_to_first_N_words(text,N=80):
    words = text.split()
    limited_words = words[:N]
    return ' '.join(limited_words)

class CustomRetriever_DuckDuckGo(VectorStoreRetriever):
    search_kwargs: dict = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True  # Allow custom types

    def __init__(self, vectorstore, search_kwargs, **data):
        super().__init__(vectorstore=vectorstore, retriever=None, **data)
        self.vectorstore = vectorstore
        self.search_kwargs = search_kwargs

    def get_relevant_documents(self, query: str) -> List[Document]:
        top_docs = enhanced_retriever_duckduckgo(
            query, max_return_docs=max_return_docs)
        return top_docs

class CustomRetriever_Tavily(BaseRetriever):
    search_kwargs: dict = Field(default_factory=dict)
    class Config:
        arbitrary_types_allowed = True  # Allow custom types
    def __init__(self, search_kwargs, **data):
        super().__init__(retriever=None, **data)
        self.search_kwargs = search_kwargs
    def get_relevant_documents(self, query: str) -> List[Document]:
        top_docs = enhanced_retriever_tavily(
            query, max_return_docs=max_return_docs)
        return top_docs

def get_best_VecDB_info(VecDB_TYPE_ACTUAL,
                        st,
                        max_return_docs,
                        ensemble_retriever_bm25_relative_weight,
                        ensemble_retriever_FAISS_relative_weight,
                        ensemble_retriever_DuckDuckGo_relative_weight,
                        enable_context_llm_filtering,
                        user_input):
    # sanity
    faiss_persist_directory = f"./faiss/{VecDB_TYPE_ACTUAL}"
    if 'vectorstore' not in st.session_state or 'vectorstore_docs' not in st.session_state:
        print(f"FAISS PERSIST DIRECTORY:{faiss_persist_directory}")
        print("INITIAL LOADING of  vectorstores & vectorstore_docs \n")
        st.session_state.vectorstore = load_faiss_vectorstore(
            faiss_persist_directory)
        # st.session_state.vectorstore.persist()
        st.session_state.vectorstore_docs = get_docs_from_faiss_vectorstore(
            st.session_state.vectorstore)    

    if configs['generate']['enable_dynamic_vec_db_selection']:
        print("ENABLE_DYNAMIC_VEC_DB_SELECTION set to True \n")
        print("Switching vectorstores \n")
        st.session_state.vectorstore = load_faiss_vectorstore(
            faiss_persist_directory)
        # st.session_state.vectorstore.persist()
        st.session_state.vectorstore_docs = get_docs_from_faiss_vectorstore(
            st.session_state.vectorstore)

    print(f"{faiss_persist_directory} vectorstore loaded.")
    # get retreivers
    # REF: https://python.langchain.com/docs/modules/data_connection/retrievers/ensemble
    # st.session_state.bm25_retriever = BM25Retriever.from_documents(st.session_state.vectorstore_docs)
    # st.session_state.bm25_retriever.k = max_return_docs
    st.session_state.bm25_retriever = CustomRetriever_BM25(
        vectorstore=st.session_state.vectorstore, search_kwargs={"k": max_return_docs})
    # INITIAL CHROMA retriever
    # st.session_state.chroma_retriever = st.session_state.vectorstore.as_retriever(search_type=search_type, search_kwargs={"k": initial_retrieval_count})
    # MODIFIED CHROMA retriever
    # st.session_state.chroma_retriever = CustomRetriever(vectorstore=st.session_state.vectorstore,search_type="mmr",search_kwargs={"k": max_return_docs})
   # FAISS retriever
    st.session_state.FAISS_retriever = CustomRetriever_FAISS(
        vectorstore=st.session_state.vectorstore, search_kwargs={"k": max_return_docs})
   # DuckDuckGo retreiver
    st.session_state.DuckDuckGo_retriever = CustomRetriever_DuckDuckGo(
        vectorstore=st.session_state.vectorstore, search_kwargs={"k": max_return_docs})
  # Tavily retreiver
    st.session_state.Tavily_retriever = CustomRetriever_Tavily(
         search_kwargs={"k": max_return_docs})    
    if configs['retrieval']['disable_llm_embed_retreiver']:
        st.session_state.chroma_retriever = st.session_state.bm25_retriever
    # initialize the ensemble retriever
    st.session_state.retriever = EnsembleRetriever(
        retrievers=[st.session_state.bm25_retriever, st.session_state.FAISS_retriever, st.session_state.DuckDuckGo_retriever], weights=[
            ensemble_retriever_bm25_relative_weight, ensemble_retriever_FAISS_relative_weight, ensemble_retriever_DuckDuckGo_relative_weight]
    )
    # Update EnsembleRetriever
    # Retriver with Contextual compression
    if enable_context_llm_filtering == 1:
        compressor = LLMChainFilter.from_llm(
            st.session_state.llm, custom_llm_chain_filter_prompt_template)
        st.session_state.retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=st.session_state.retriever
        )

    st.session_state.qa_chain = RetrievalQA.from_chain_type(
        llm=st.session_state.llm,
        chain_type='stuff',
        # ACTUAL ONE THAT WORKS (only semantic retrieval no TF-IDF)
        retriever=st.session_state.retriever,
        verbose=configs['retrieval']['enable_verbose'],
        chain_type_kwargs={
            "verbose": configs['retrieval']['enable_verbose'],
            "prompt": st.session_state.prompt,
            "memory": st.session_state.memory,
        }
    )
    return st.session_state.bm25_retriever, st.session_state.retriever, st.session_state.qa_chain, st.session_state.FAISS_retriever, st.session_state.DuckDuckGo_retriever, st.session_state.Tavily_retriever





def combine_langchain_documents(doc_list):
# Usage example:
# docs = [doc1, doc2, doc3, doc4]  # any number of documents
# combined_doc = combine_documents(docs)    
    # Combine content with newline separation
    combined_content = "\n".join(doc.page_content for doc in doc_list)    
    # Collect all sources from metadata
    combined_metadata = {
        "sources": [doc.metadata.get("source", "") for doc in doc_list]
    }    
    # Create new combined document
    return Document(
        page_content=combined_content,
        metadata=combined_metadata
    )

    



def add_fake_id_to_documents(documents):
    """
    Takes a list of LangChain Document objects that lack an 'id' attribute
    and returns a new list with a generated UUID (as a string) for each doc.
    """
    updated_documents = []
    for doc in documents:
        # Generate a unique ID (UUIDv4)
        new_id = str(uuid.uuid4())        
        # Create a new Document with the same page_content and metadata,
        # but add an 'id' attribute
        updated_doc = Document(
            page_content=doc.page_content,
            metadata=doc.metadata,
            id=new_id
        )        
        updated_documents.append(updated_doc)    
    return updated_documents    


# heyligo_search
def heyligo_search(query, sources=['LLO', 'LHO', 'DCC'], fetch_content=True, max_results=5, verbose=False):
    """
    Performs search and returns documents and combined text.

    Parameters:
        query (str): The search query.
        sources (list): List of sources to include. Default is ['LLO', 'LHO', 'DCC'].
        fetch_content (bool): If True, fetch and include the page content for each result.
        max_results (int or dict): Maximum number of results per source. Default is None.
        verbose (bool): If True, prints progress messages.

    Returns:
        tuple: (documents, combined_text)
            documents (list): A list of LangChain Document objects.
            combined_text (str): Combined text of all results.
    """

    # Helper function to extract HTML text
    def extract_html_text(url):
        headers = {
            'User-Agent': 'Mozilla/5.0',  # Mimic a browser
        }
        session = requests.Session()
        try:
            response = session.get(url, headers=headers)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            if verbose:
                print(f"Failed to fetch content from {url}. Error: {e}")
            return ''

        # Parse the main page HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the iframe
        iframe = soup.find('iframe')
        if iframe and iframe.has_attr('src'):
            iframe_src = iframe['src']
            # If the iframe src is a relative URL, construct the full URL
            iframe_url = urljoin(url, iframe_src)
            if verbose:
                print(
                    f"Found iframe. Fetching content from iframe URL: {iframe_url}")
            try:
                response_iframe = session.get(iframe_url, headers=headers)
                response_iframe.raise_for_status()
            except requests.exceptions.RequestException as e:
                if verbose:
                    print(
                        f"Failed to fetch iframe content from {iframe_url}. Error: {e}")
                return ''
            # Parse the iframe content
            soup_iframe = BeautifulSoup(response_iframe.content, 'html.parser')
            # Remove script and style elements
            for script_or_style in soup_iframe(['script', 'style', 'noscript']):
                script_or_style.decompose()
            # Get text
            text = soup_iframe.get_text(separator='\n')
        else:
            # No iframe found, process the main page content
            # Remove script and style elements
            for script_or_style in soup(['script', 'style', 'noscript']):
                script_or_style.decompose()
            # Get text
            text = soup.get_text(separator='\n')

        # Clean up the text
        lines = [line.strip() for line in text.splitlines()]
        text = '\n'.join(line for line in lines if line)
        if verbose:
            print(f"Text extraction complete for URL: {url}")
        return text

    if verbose:
        print(f"Starting search with query: '{query}'")
        print(f"Sources selected: {sources}")
        print(f"Fetch content: {fetch_content}")
        print(f"Max results per source: {max_results}")

    # Mapping of sources to their endpoints and URL patterns
    source_info = {
        'LLO': {
            'endpoint': 'http://heyligo.gw.iucaa.in/lsearch_authors_titles',
            'url_pattern': 'https://alog.ligo-la.caltech.edu/aLOG/iframeSrc.php?callRep={callRep}',
        },
        'LHO': {
            'endpoint': 'http://heyligo.gw.iucaa.in/hsearch_authors_titles',
            'url_pattern': 'https://alog.ligo-wa.caltech.edu/aLOG/iframeSrc.php?callRep={callRep}',
        },
    }

    headers = {
        'User-Agent': 'Mozilla/5.0',  # Optional: Mimic a browser user agent
    }

    results = []
    documents = []
    dcc_results_collected = False  # Flag to avoid fetching DCC results multiple times

    for source in ['LLO', 'LHO']:
        if source in sources:
            if verbose:
                print(f"\nFetching data from source: {source}")
            info = source_info[source]
            url = info['endpoint']
            url_pattern = info['url_pattern']
            data = {'term': query}

            try:
                response = requests.post(url, data=data, headers=headers)
                response.raise_for_status()
                if verbose:
                    print(f"Successfully fetched data from {source}")
            except requests.exceptions.RequestException as e:
                if verbose:
                    print(f"Failed to fetch data from {source}. Error: {e}")
                continue

            json_data = response.json()
            source_results = []

            # Extract top searches (aLOG entries)
            top_searches = json_data.get('top_searches', [])
            if verbose:
                print(
                    f"Number of aLOG entries retrieved from {source}: {len(top_searches)}")

            # Limit the number of results per source
            if max_results:
                if isinstance(max_results, dict):
                    limit = max_results.get(source, None)
                else:
                    limit = max_results
                if limit is not None:
                    top_searches = top_searches[:limit]
                    if verbose:
                        print(
                            f"Limiting results for {source} to top {limit} entries")

            for item in top_searches:
                callRep = item[0]
                title = item[1]
                summary = item[2]
                date_str = item[3]
                # Parse the date string into a datetime object
                try:
                    date = parse_date(date_str.strip())
                except (ValueError, OverflowError) as e:
                    if verbose:
                        print(
                            f"Failed to parse date '{date_str}' for entry '{title}'. Error: {e}")
                    date = None  # Handle invalid date formats

                url_alog = url_pattern.format(callRep=callRep)
                result_entry = {
                    'source': source,
                    'type': 'alog',
                    'title': title.strip(),
                    'summary': summary.strip(),
                    'date': date,
                    'url': url_alog
                }

                source_results.append(result_entry)

            # Fetch content if enabled, after limiting results
            if fetch_content:
                for result_entry in source_results:
                    if verbose:
                        print(
                            f"Fetching content for URL: {result_entry['url']}")
                    content = extract_html_text(result_entry['url'])
                    result_entry['content'] = content
                    if verbose:
                        print(
                            f"Content fetched for URL: {result_entry['url']}")

                    # Create a LangChain Document
                    doc = Document(
                        page_content=content,
                        metadata={
                            'source': result_entry['source'],
                            'type': result_entry['type'],
                            'title': result_entry['title'],
                            'date': result_entry['date'],
                            'url': result_entry['url'],
                        }
                    )
                    documents.append(doc)
            else:
                # Even if content is not fetched, create Document with empty content
                for result_entry in source_results:
                    doc = Document(
                        page_content='',
                        metadata={
                            'source': result_entry['source'],
                            'type': result_entry['type'],
                            'title': result_entry['title'],
                            'date': result_entry['date'],
                            'url': result_entry['url'],
                        }
                    )
                    documents.append(doc)

            # Sort the source results by date (latest first)
            source_results = sorted(
                source_results,
                key=lambda x: x['date'] if x['date'] else datetime.min,
                reverse=True
            )

            results.extend(source_results)
            if verbose:
                print(f"Total results from {source}: {len(source_results)}")

            # Extract DCC entries only once
            if 'DCC' in sources and not dcc_results_collected:
                if verbose:
                    print("Fetching DCC entries")
                dcc_results = json_data.get('dcc_result', [])
                if verbose:
                    print(
                        f"Number of DCC entries retrieved: {len(dcc_results)}")

                # Limit the number of DCC results
                if max_results:
                    if isinstance(max_results, dict):
                        limit = max_results.get('DCC', None)
                    else:
                        limit = max_results
                    if limit is not None:
                        dcc_results = dcc_results[:limit]
                        if verbose:
                            print(
                                f"Limiting DCC results to top {limit} entries")

                dcc_source_results = []

                for item in dcc_results:
                    title = item[0]
                    summary = item[1]
                    url_path = item[2]
                    date_str = item[3]
                    try:
                        date = parse_date(date_str.strip())
                    except (ValueError, OverflowError) as e:
                        if verbose:
                            print(
                                f"Failed to parse date '{date_str}' for DCC entry '{title}'. Error: {e}")
                        date = None

                    url_dcc = f"https://dcc.ligo.org{url_path}"
                    result_entry = {
                        'source': 'DCC',
                        'type': 'dcc',
                        'title': title.strip(),
                        'summary': summary.strip(),
                        'date': date,
                        'url': url_dcc
                    }

                    dcc_source_results.append(result_entry)

                # Fetch content if enabled, after limiting results
                if fetch_content:
                    for result_entry in dcc_source_results:
                        if verbose:
                            print(
                                f"Fetching content for DCC URL: {result_entry['url']}")
                        content = extract_html_text(result_entry['url'])
                        result_entry['content'] = content
                        if verbose:
                            print(
                                f"Content fetched for DCC URL: {result_entry['url']}")

                        # Create a LangChain Document
                        doc = Document(
                            page_content=content,
                            metadata={
                                'source': result_entry['source'],
                                'type': result_entry['type'],
                                'title': result_entry['title'],
                                'date': result_entry['date'],
                                'url': result_entry['url'],
                            }
                        )
                        documents.append(doc)
                else:
                    # Create Document with empty content
                    for result_entry in dcc_source_results:
                        doc = Document(
                            page_content='',
                            metadata={
                                'source': result_entry['source'],
                                'type': result_entry['type'],
                                'title': result_entry['title'],
                                'date': result_entry['date'],
                                'url': result_entry['url'],
                            }
                        )
                        documents.append(doc)

                # Sort DCC results by date
                dcc_source_results = sorted(
                    dcc_source_results,
                    key=lambda x: x['date'] if x['date'] else datetime.min,
                    reverse=True
                )

                results.extend(dcc_source_results)
                if verbose:
                    print(f"Total DCC results: {len(dcc_source_results)}")
                dcc_results_collected = True  # Avoid fetching DCC results again

    # If only 'DCC' is selected and neither 'LLO' nor 'LHO', fetch DCC results from one endpoint
    if 'DCC' in sources and not ('LLO' in sources or 'LHO'):
        if verbose:
            print("Only DCC selected. Fetching DCC entries from LLO endpoint.")
        info = source_info['LLO']
        url = info['endpoint']
        data = {'term': query}

        try:
            response = requests.post(url, data=data, headers=headers)
            response.raise_for_status()
            if verbose:
                print("Successfully fetched data for DCC from LLO endpoint")
        except requests.exceptions.RequestException as e:
            if verbose:
                print(f"Failed to fetch DCC data. Error: {e}")
            return documents, ''

        json_data = response.json()
        dcc_results = json_data.get('dcc_result', [])
        if verbose:
            print(f"Number of DCC entries retrieved: {len(dcc_results)}")

        # Limit the number of DCC results
        if max_results:
            if isinstance(max_results, dict):
                limit = max_results.get('DCC', None)
            else:
                limit = max_results
            if limit is not None:
                dcc_results = dcc_results[:limit]
                if verbose:
                    print(f"Limiting DCC results to top {limit} entries")

        dcc_source_results = []

        for item in dcc_results:
            title = item[0]
            summary = item[1]
            url_path = item[2]
            date_str = item[3]
            try:
                date = parse_date(date_str.strip())
            except (ValueError, OverflowError) as e:
                if verbose:
                    print(
                        f"Failed to parse date '{date_str}' for DCC entry '{title}'. Error: {e}")
                date = None

            url_dcc = f"https://dcc.ligo.org{url_path}"
            result_entry = {
                'source': 'DCC',
                'type': 'dcc',
                'title': title.strip(),
                'summary': summary.strip(),
                'date': date,
                'url': url_dcc
            }

            dcc_source_results.append(result_entry)

        # Fetch content if enabled, after limiting results
        if fetch_content:
            for result_entry in dcc_source_results:
                if verbose:
                    print(
                        f"Fetching content for DCC URL: {result_entry['url']}")
                content = extract_html_text(result_entry['url'])
                result_entry['content'] = content
                if verbose:
                    print(
                        f"Content fetched for DCC URL: {result_entry['url']}")

                # Create a LangChain Document
                doc = Document(
                    page_content=content,
                    metadata={
                        'source': result_entry['source'],
                        'type': result_entry['type'],
                        'title': result_entry['title'],
                        'date': result_entry['date'],
                        'url': result_entry['url'],
                    }
                )
                documents.append(doc)
        else:
            # Create Document with empty content
            for result_entry in dcc_source_results:
                doc = Document(
                    page_content='',
                    metadata={
                        'source': result_entry['source'],
                        'type': result_entry['type'],
                        'title': result_entry['title'],
                        'date': result_entry['date'],
                        'url': result_entry['url'],
                    }
                )
                documents.append(doc)

        # Sort DCC results by date
        dcc_source_results = sorted(
            dcc_source_results,
            key=lambda x: x['date'] if x['date'] else datetime.min,
            reverse=True
        )

        results.extend(dcc_source_results)
        if verbose:
            print(f"Total DCC results: {len(dcc_source_results)}")

    if verbose:
        print("\nSearch completed.")
        print(f"Total results found: {len(results)}")

    # Generate combined text
    combined_text = ''
    separator = '------------------------\n'

    for result in results:
        date_str = result['date'].strftime(
            '%Y-%m-%d %H:%M:%S') if result['date'] else 'Unknown Date'
        content = result.get('content', '')
        result_text = (
            f"Source: {result['source']}\n"
            f"Type: {result['type']}\n"
            f"Title: {result['title']}\n"
            f"Date: {date_str}\n"
            f"URL: {result['url']}\n"
        )
        if content:
            result_text += f"Content:\n{content}\n"
        result_text += separator
        combined_text += result_text

    return documents, combined_text


"""
Standalone Hey LIGO search using Selenium
========================================

This module provides a self‑contained implementation of
``heyligo_selenium_search`` that reproduces the behaviour of the original
function but uses Selenium to drive the Hey LIGO web interface.
It does not import any helper functions from other modules in this
repository; all necessary helpers (page fetching, result parsing
and search logic) are defined within this file.

The signature is compatible with the legacy ``heyligo_selenium_search``::

    def heyligo_selenium_search(query, sources=['LLO','LHO','DCC'],
                       fetch_content=True, max_results=5, verbose=False)
        -> Tuple[List[Document], str]

``Document`` refers to ``langchain.schema.Document`` if available;
otherwise a minimal stand‑in class with the same interface is used.

To use this module you must have Selenium installed and a
corresponding WebDriver (e.g. ChromeDriver) available on your PATH.

Example usage::

    docs, text = heyligo_selenium_search(
        query="OMC scan",
        sources=["LLO", "DCC"],
        fetch_content=True,
        max_results={"LLO": 5, "DCC": 3},
        verbose=True,
    )
"""

# Attempt to import Document from langchain; provide a fallback if unavailable
try:
    from langchain.schema import Document  # type: ignore
except Exception:
    class Document:
        """Minimal stand‑in for langchain.schema.Document.

        This class mimics LangChain's Document interface enough for
        consuming code.  It stores the ``page_content`` and ``metadata``
        and exposes them as attributes.
        """

        def __init__(self, page_content: str, metadata: Dict[str, Any]) -> None:
            self.page_content = page_content
            self.metadata = metadata

# Attempt to import dateutil.parser.parse for flexible date parsing
try:
    from dateutil.parser import parse as parse_date  # type: ignore
except Exception:
    parse_date = None  # type: ignore


def fetch_page_text(url: str, *, ssl_verify: bool = True) -> Dict[str, Any]:
    """Fetch a web page and extract plain text.

    This helper uses ``requests`` to retrieve a page and ``BeautifulSoup``
    to parse the HTML.  It removes ``<script>``, ``<style>`` and
    ``<noscript>`` elements before extracting the text.  The first
    non‑empty line is treated as the header (title) of the page.

    Parameters
    ----------
    url : str
        The URL of the page to fetch.
    ssl_verify : bool, optional
        Whether to verify TLS certificates.  Defaults to True.

    Returns
    -------
    dict
        A dictionary with keys ``url``, ``header`` and ``text``.
    """
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, verify=ssl_verify)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    header = text.split("\n", 1)[0] if text else ""
    return {"url": url, "header": header, "text": text}


def selenium_search_with_dcc(
    query: str,
    *,
    site: str = "LLO",
    max_links: Optional[int] = None,
    max_dcc_links: Optional[int] = None,
    headless: bool = True,
    ssl_verify: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """Perform a Hey LIGO search via Selenium and scrape aLOG and DCC results.

    Parameters
    ----------
    query : str
        The search query to submit.
    site : {"LLO", "LHO"}, optional
        Which observatory's logbook to search.  Default is "LLO".
    max_links : int or None, optional
        Limit on the number of aLOG results.  ``None`` means no explicit limit.
    max_dcc_links : int or None, optional
        Limit on the number of DCC results.  ``None`` means no explicit limit.
    headless : bool, optional
        Whether to run the browser in headless mode.  Default True.
    ssl_verify : bool, optional
        Whether to verify TLS certificates when fetching pages.  Default True.

    Returns
    -------
    dict
        A dictionary with two keys: ``"alog"`` and ``"dcc"``.  Each maps
        to a list of entry dictionaries containing ``url``, ``header`` and
        ``text``.
    """
    ui_paths = {
        "LLO": "https://heyligo.gw.iucaa.in/lauthors_titles",
        "LHO": "https://heyligo.gw.iucaa.in/hauthors_titles",
    }
    if site not in ui_paths:
        raise ValueError("site must be 'LLO' or 'LHO'")
    page_url = ui_paths[site]
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    alog_entries: List[Dict[str, Any]] = []
    dcc_entries: List[Dict[str, Any]] = []
    try:
        driver.get(page_url)
        wait = WebDriverWait(driver, 30)
        search_box = wait.until(EC.element_to_be_clickable((By.ID, "srch-term")))
        search_box.clear()
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)
        # Wait for results to appear in either RelatedPost or DCC
        wait.until(
            EC.any_of(
                EC.presence_of_element_located((By.CSS_SELECTOR, "#RelatedPost a")),
                EC.presence_of_element_located((By.CSS_SELECTOR, "#DCC a")),
            )
        )
        # Collect aLOG links
        alog_anchors = driver.find_elements(By.CSS_SELECTOR, "#RelatedPost a")
        for anchor in alog_anchors[: (max_links or len(alog_anchors))]:
            href = anchor.get_attribute("href")
            if not href:
                continue
            try:
                alog_entries.append(fetch_page_text(href, ssl_verify=ssl_verify))
            except Exception:
                # Skip entries that fail to fetch
                continue
        # Collect DCC links
        dcc_anchors = driver.find_elements(By.CSS_SELECTOR, "#DCC a")
        for anchor in dcc_anchors[: (max_dcc_links or len(dcc_anchors))]:
            href = anchor.get_attribute("href")
            if not href:
                continue
            try:
                dcc_entries.append(fetch_page_text(href, ssl_verify=ssl_verify))
            except Exception:
                continue
    finally:
        driver.quit()
    return {"alog": alog_entries, "dcc": dcc_entries}


def _guess_datetime_from_header(header: str) -> Optional[datetime]:
    """Attempt to parse a date from a log entry header.

    aLOG headers typically contain a date in the format
    ``"Wednesday 08 May 2019"``.  This helper extracts the date
    substring and uses ``dateutil.parser.parse`` if available.  If
    parsing fails, returns ``None``.
    """
    # Regex to capture day month year (with optional weekday)
    date_pattern = re.compile(
        r"(?:\b(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day\s+)?(\d{1,2}\s+\w+\s+\d{4})",
        re.IGNORECASE,
    )
    match = date_pattern.search(header)
    if match and parse_date is not None:
        date_str = match.group(1)
        try:
            return parse_date(date_str)
        except Exception:
            return None
    return None


def heyligo_selenium_search(
    query: str,
    sources: List[str] | None = None,
    fetch_content: bool = True,
    max_results: Union[int, Dict[str, int], None] = 5,
    verbose: bool = False,
) -> Tuple[List[Document], str]:
    """Search Hey LIGO and return documents and combined text.

    This function mimics the signature of the legacy
    ``heyligo_selenium_search`` but uses Selenium internally.  It can
    search the Livingston (LLO) and Hanford (LHO) logbooks and optionally
    include DCC entries.

    Parameters
    ----------
    query : str
        The search query to submit.
    sources : list of str, optional
        Which sources to include.  Valid options are 'LLO', 'LHO' and
        'DCC'.  Defaults to all three if ``None``.
    fetch_content : bool, optional
        If ``True``, fetch and include the full text of each result.
        If ``False``, the Document content will be empty.  Default True.
    max_results : int or dict or None, optional
        Limit on the number of results per source.  If an integer is
        provided, it applies uniformly.  If a dict is given, keys may
        specify limits for 'LLO', 'LHO' and 'DCC' separately.  ``None``
        means no limit.  Default 5.
    verbose : bool, optional
        If ``True``, print progress messages.  Default False.

    Returns
    -------
    tuple
        ``(documents, combined_text)`` where ``documents`` is a list of
        ``Document`` objects (or their fallback) and ``combined_text``
        concatenates the formatted results.
    """
    if sources is None:
        sources = ["LLO", "LHO", "DCC"]
    valid_sources = {"LLO", "LHO", "DCC"}
    for s in sources:
        if s not in valid_sources:
            raise ValueError(f"Unknown source '{s}'. Valid options are {valid_sources}.")

    def limit_for(name: str) -> Optional[int]:
        if max_results is None:
            return None
        if isinstance(max_results, dict):
            return max_results.get(name)
        return max_results  # assume int

    documents: List[Document] = []
    results: List[Dict[str, Any]] = []
    seen_dcc_urls: set[str] = set()
    dcc_collected = False

    # Process aLOG sources in a fixed order
    for site in ["LLO", "LHO"]:
        if site not in sources:
            continue
        log_limit = limit_for(site)
        if "DCC" in sources and not dcc_collected:
            dcc_limit = limit_for("DCC")
        else:
            dcc_limit = 0
        if verbose:
            print(f"Querying {site} (log limit={log_limit}, dcc limit={dcc_limit})")
        try:
            search_data = selenium_search_with_dcc(
                query,
                site=site,
                max_links=log_limit,
                max_dcc_links=dcc_limit,
                headless=True,
                ssl_verify=True,
            )
        except Exception as exc:
            if verbose:
                print(f"Error querying {site}: {exc}")
            continue
        # Handle aLOG entries
        for entry in search_data["alog"]:
            content = entry["text"] if fetch_content else ""
            date = _guess_datetime_from_header(entry["header"])
            # Shorten the aLOG URL for LLO/LHO
            alog_url = entry["url"]
            try:
                parsed = urllib.parse.urlparse(alog_url)
                qs = urllib.parse.parse_qs(parsed.query)
                callrep_values = qs.get("callRep") or qs.get("callrep")
                if callrep_values:
                    callrep = callrep_values[0]
                    domain = "ligo-la.caltech.edu" if "ligo-la" in alog_url else "ligo-wa.caltech.edu"
                    alog_url_short = f"https://alog.{domain}/aLOG/index.php?callRep={callrep}"
                else:
                    alog_url_short = alog_url
            except Exception:
                alog_url_short = alog_url
            # Inject URL into content
            content_with_url = f"{content}\n\nURL: {alog_url_short}" if fetch_content else ""
            results.append(
                {
                    "source": site,
                    "type": "alog",
                    "title": entry["header"],
                    "summary": "",
                    "date": date,
                    "url": alog_url_short,
                    "content": content_with_url,
                }
            )
            documents.append(
                Document(
                    page_content=content_with_url,
                    metadata={
                        "source": site,
                        "type": "alog",
                        "title": entry["header"],
                        "date": date,
                        "url": alog_url_short,
                    },
                )
            )
        # Handle DCC entries if requested and not already collected
        if "DCC" in sources and not dcc_collected:
            for entry in search_data["dcc"]:
                if entry["url"] in seen_dcc_urls:
                    continue
                seen_dcc_urls.add(entry["url"])
                content = entry["text"] if fetch_content else ""
                content_with_url = f"{content}\n\nURL: {entry['url']}" if fetch_content else ""
                results.append(
                    {
                        "source": "DCC",
                        "type": "dcc",
                        "title": entry["header"],
                        "summary": "",
                        "date": None,
                        "url": entry["url"],
                        "content": content_with_url,
                    }
                )
                documents.append(
                    Document(
                        page_content=content_with_url,
                        metadata={
                            "source": "DCC",
                            "type": "dcc",
                            "title": entry["header"],
                            "date": None,
                            "url": entry["url"],
                        },
                    )
                )
            dcc_collected = True

    # Sort results by date if available, descending
    results_sorted = sorted(
        results,
        key=lambda x: x["date"] if x["date"] is not None else datetime.min,
        reverse=True,
    )
    # Build combined text
    combined_text = ""
    separator = "------------------------\n"
    for r in results_sorted:
        date_str = r["date"].strftime("%Y-%m-%d %H:%M:%S") if r["date"] else "Unknown Date"
        entry_text = (
            f"Source: {r['source']}\n"
            f"Type: {r['type']}\n"
            f"Title: {r['title']}\n"
            f"Date: {date_str}\n"
            f"URL: {r['url']}\n"
        )
        if fetch_content and r["content"]:
            entry_text += f"Content:\n{r['content']}\n"
        entry_text += separator
        combined_text += entry_text
    if verbose:
        print(f"Search completed. Total results: {len(results_sorted)}")
    return documents, combined_text

def extract_text_from_url(url):
    """
    Extracts plain text from a web page, handling HTML tags.    
    Args:
        url (str): The URL of the web page to extract text from.    
    Returns:
        str: The extracted text with HTML tags removed.
        None: If an error occurs during the process.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes        
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')        
        # Remove script and style elements
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()        
        # Get text from the parsed HTML
        text = soup.get_text()        
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())        
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))        
        # Remove blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)        
        return text
    except requests.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    

def extract_url_from_Tavily_Doc_source_metadata(doc):
    try:
        return doc.metadata.get('source')
    except AttributeError:
        return None
    




def fetch_via_pdfminer_data_from_url(url):
    response = requests.get(url)
    pdf_bytes = BytesIO(response.content)
    text = extract_text(pdf_bytes)
    return text



PDF_RE = re.compile(r"\.pdf($|\?)", re.I)        # ".pdf" before end or "?"
TIMEOUT = 15

def _download_pdf_text(pdf_url,timeout=TIMEOUT):
    """Helper – download one PDF and return text (empty string on failure)."""
    try:
        r = requests.get(pdf_url, timeout=TIMEOUT)
        r.raise_for_status()
        return extract_text(BytesIO(r.content))
    except Exception as e:
        print(f"[PDF-fetch] {pdf_url} – {e}")
        return ""

def get_text_from_url_including_pdf(url,max_tokens,timeout=TIMEOUT):
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        ctype = resp.headers.get("content-type", "").lower()

        # ───────────────────────── DIRECT PDF ─────────────────────────────
        if url.lower().endswith(".pdf") or "application/pdf" in ctype:
            page_text = extract_text(BytesIO(resp.content))

        # ─────────────────────────  HTML  ─────────────────────────────────
        else:
            soup = BeautifulSoup(resp.content, "html.parser")
            for s in soup(["script", "style"]): s.decompose()

            # page’s own visible text
            text_lines = (ln.strip() for ln in soup.get_text().splitlines())
            chunks     = (ck.strip() for ln in text_lines for ck in ln.split("  "))
            page_text  = "\n".join(ck for ck in chunks if ck)

            # ── find PDF-like links ───────────────────────────────────────
            pdf_texts = []
            for tag in soup.find_all("a", href=True):
                href = tag["href"]
                if not PDF_RE.search(href):          # skip non-PDF links
                    continue

                first_hop = urljoin(url, href)       # absolute URL

                # Some DCC links lead to a “file-card” HTML page; detect that
                # and, if so, grab the final ?download link inside it.
                if not first_hop.lower().endswith(".pdf") and "application/pdf" not in ctype:
                    try:
                        card = requests.get(first_hop, timeout=timeout)
                        card.raise_for_status()
                        card_soup = BeautifulSoup(card.content, "html.parser")
                        dl_tag = card_soup.find("a", href=PDF_RE)
                        if dl_tag:
                            first_hop = urljoin(first_hop, dl_tag["href"])
                    except Exception as ee:
                        print(f"[card-hop] {first_hop} – {ee}")

                pdf_txt = _download_pdf_text(first_hop,timeout=timeout)
                if pdf_txt:
                    pdf_texts.append(f"\n\n----- PDF ({first_hop}) -----\n{pdf_txt}")

            page_text += "".join(pdf_texts)

        # ─────────────────── global truncation ────────────────────────────
        if max_tokens is not None and max_tokens > 0:
            tokens = page_text.split()
            page_text = " ".join(tokens[:max_tokens])

        return page_text

    except Exception as exc:
        print(f"[get_text_from_url] {url} – {exc}")
        return None





def generate_wiki_summary_from_langchain_docs(
    llm,
    langchain_docs: List[Any],
    user_query: str,
    *,
    log_skipped: bool = True,
) -> str:
    """
    Safely build a Wikipedia-style summary from a heterogeneous list of
    LangChain Documents.  Non-Document items are ignored.
    """
    # ── 1. keep only objects that look like Documents ────────────────
    clean_docs = []
    skipped = 0
    for d in langchain_docs:
        if hasattr(d, "page_content") and hasattr(d, "metadata"):
            clean_docs.append(d)
        else:
            skipped += 1
            if log_skipped:
                print(f"[generate_wiki_summary] skipped non-Document: {type(d)}")
    if skipped and log_skipped:
        print(f"[generate_wiki_summary] accepted={len(clean_docs)} skipped={skipped}")
    # ── 2. short-circuit if nothing to summarise ─────────────────────
    if not clean_docs:
        return ""   # or "(no documents)" if you prefer a visible placeholder
    # ── 3. build the long prompt text  ───────────────────────────────
    separator = "\n\n ---------------- \n\n"
    ans_list_initial = [
        f"{doc.page_content} | Metadata: {doc.metadata}" for doc in clean_docs
    ]
    ans_list_initial_combined = separator.join(ans_list_initial)
    tmpl = configs["prompts"]["WIKI_SUMMARY_template"]["template"]
    prompt = PromptTemplate(
        input_variables=["summary_text", "user_query"], template=tmpl
    ).format(summary_text=ans_list_initial_combined, user_query=user_query)
    # ── 4. call the LLM  ─────────────────────────────────────────────
    with st.spinner(f"Summarising {len(clean_docs)} retrieved & filtered documents…"):
        raw_out = llm(prompt)
    # ── 5. extract between <ARTICLE> … </ARTICLE> if present ─────────
    matches = re.findall(r"<ARTICLE>(.*?)</ARTICLE>", raw_out, re.DOTALL)
    wiki_summary = " \n".join(matches) if matches else raw_out
    # ── 6. post-processing helpers you already have ──────────────────
    wiki_summary = wrap_urls_in_angle_brackets(wiki_summary)
    wiki_summary = replace_plus_in_url_paths(wiki_summary)
    wiki_summary = dedup_paragraphs(wiki_summary)
    return wiki_summary


def generate_wiki_summary_from_langchain_docs_api_version(
    llm,
    langchain_docs: List[Any],
    user_query: str,
    *,
    log_skipped: bool = True,
) -> str:
    """
    Safely build a Wikipedia-style summary from a heterogeneous list of
    LangChain Documents.  Non-Document items are ignored.
    """
    # ── 1. keep only objects that look like Documents ────────────────
    clean_docs = []
    skipped = 0
    for d in langchain_docs:
        if hasattr(d, "page_content") and hasattr(d, "metadata"):
            clean_docs.append(d)
        else:
            skipped += 1
            if log_skipped:
                print(f"[generate_wiki_summary] skipped non-Document: {type(d)}")
    if skipped and log_skipped:
        print(f"[generate_wiki_summary] accepted={len(clean_docs)} skipped={skipped}")
    # ── 2. short-circuit if nothing to summarise ─────────────────────
    if not clean_docs:
        return ""   # or "(no documents)" if you prefer a visible placeholder
    # ── 3. build the long prompt text  ───────────────────────────────
    separator = "\n\n ---------------- \n\n"
    ans_list_initial = [
        f"{doc.page_content} | Metadata: {doc.metadata}" for doc in clean_docs
    ]
    ans_list_initial_combined = separator.join(ans_list_initial)
    tmpl = configs["prompts"]["WIKI_SUMMARY_template"]["template"]
    prompt = PromptTemplate(
        input_variables=["summary_text", "user_query"], template=tmpl
    ).format(summary_text=ans_list_initial_combined, user_query=user_query)
    # ── 4. call the LLM  ─────────────────────────────────────────────
    #with st.spinner(f"Summarising {len(clean_docs)} retrieved & filtered documents…"):
    raw_out = llm(prompt)
    # ── 5. extract between <ARTICLE> … </ARTICLE> if present ─────────
    matches = re.findall(r"<ARTICLE>(.*?)</ARTICLE>", raw_out, re.DOTALL)
    wiki_summary = " \n".join(matches) if matches else raw_out
    # ── 6. post-processing helpers you already have ──────────────────
    wiki_summary = wrap_urls_in_angle_brackets(wiki_summary)
    wiki_summary = replace_plus_in_url_paths(wiki_summary)
    wiki_summary = dedup_paragraphs(wiki_summary)
    return wiki_summary


def ensembele_superposition_answer_from_docs(filtered_retrieved_docs_LLM_answers,user_query):
    MORE_DETAILS = ""
    WIKI_SUMMARY = ""
    QA_ANSWER    = ""

    # Intialize SUMMARY
    WIKI_SUMMARY = ""
    LangGraph_SUMMARY= ""
    # Intialize QA_ANSWER_AGGREGATOR
    QA_ANSWER_AGGREGATOR = ""   

    if len(filtered_retrieved_docs_LLM_answers)>0:

        # first latexify equations
        filtered_retrieved_docs_LLM_answers = latexify_documents(filtered_retrieved_docs_LLM_answers)

        # convert to string             
        MORE_DETAILS = pretty_print_docs(filtered_retrieved_docs_LLM_answers,print_output=False)

        # Use WIKI_SUMMARY
        WIKI_SUMMARY = generate_wiki_summary_from_langchain_docs(st.session_state.llm,filtered_retrieved_docs_LLM_answers,user_query)

        # LANGGRAPH SUMMARY
        # # LangGraph_SUMMARY = ""
        # with st.spinner(f"Generating LangGraph Iterative Summary of  {len(filtered_retrieved_docs_LLM_answers)} Retrieved Documents"):    
        #     # LangGraph Summary
        #     try:
        #         #LangGraph_SUMMARY = asyncio.run(run_langgraph_summarization(st.session_state.llm,filtered_retrieved_docs_LLM_answers,user_query))        
        #         breakpoint()
        #     except:   
        #            LangGraph_SUMMARY ="Error in LangGraph_SUMMARY generation. Check Code"      
        #     # PostProcess & ensure 'URLs' are formatted correctly '<URL>'    
        #     LangGraph_SUMMARY = wrap_urls_in_angle_brackets(LangGraph_SUMMARY)    
        #     LangGraph_SUMMARY = replace_plus_in_url_paths(LangGraph_SUMMARY) 
        #     streamlit_add_line(st)
        #     streamlit_add_bold_heading(st,role="assistant-0",message="Iterative Summary")        
        #     streamlit_add_msg(st,role="assistant-0",message=LangGraph_SUMMARY)          


        # Update Summary
        # WIKI_SUMMARY = WIKI_SUMMARY + "\n--------------------\n" + "\n LangGraph Summary: \n "+LangGraph_SUMMARY

        # RAPTOR_SUMMARY = ''
        ## ######################################
        ## # RAPTOR_SUMMARY
        ## ######################################
        ## # convert to list from LangChain Docs
        ## list_filtered_retrieved_docs_LLM_answers = [f"{doc.page_content} | Metadata: {doc.metadata}" for doc in filtered_retrieved_docs_LLM_answers]
        ## 
        ## with st.spinner("Recursively Summarizing Retrieved Documents (via RAPTOR)"):
        ##     RAPTOR_SUMMARY = ""
        ##     if not len(ans_list_initial) == 0:
        ##         with st.spinner("Summarizing retrived documents"):
        ##             initial_RAPTOR_SUMMARY, RAPTOR_SUMMARY_ALL_ITERATIONS = recursive_summarization(
        ##                 list_filtered_retrieved_docs_LLM_answers, user_query, faiss_embeddings, st.session_state.llm)
        ##         # Get SUMMARY
        ##         temp_summary = " \n".join(RAPTOR_SUMMARY_ALL_ITERATIONS)
        ##     
        ##         # Use RAPTOR_SUMMARY_template
        ##         my_template = configs['prompts']['RAPTOR_SUMMARY_template']['template']
        ##         my_prompt = PromptTemplate(input_variables=["summary_text","user_query"],template=my_template)
        ##         my_formatted_prompt = my_prompt.format(summary_text=temp_summary,user_query=user_query)
        ##         my_out = st.session_state.llm(my_formatted_prompt)
        ##     
        ##         temp_out = my_out
        ##     
        ##         # Regular expression to find text between <SUMMARY> and </SUMMARY>
        ##         pattern = r'<ARTICLE>(.*?)</ARTICLE>'
        ##         # Find all matches
        ##         temp_RAPTOR_SUMMARY = re.findall(pattern, temp_out, re.DOTALL)
        ##         if len(temp_RAPTOR_SUMMARY) == 0:
        ##             RAPTOR_SUMMARY = temp_out
        ##         else:
        ##             RAPTOR_SUMMARY = " \n".join(temp_RAPTOR_SUMMARY)
        ## 

        #WIKI_SUMMARY = WIKI_SUMMARY + " \n\n " + RAPTOR_SUMMARY

        with st.spinner("Looking for the answers within the retrieved documents"):
            # Use SUPERPOSITION_QA_ANSWER_template
            my_template = configs['prompts']['SUPERPOSITION_QA_ANSWER_template']['template']
            my_prompt = PromptTemplate(input_variables=["ans_list_final","user_query"],template=my_template)
            my_formatted_prompt = my_prompt.format(ans_list_final=MORE_DETAILS,user_query=user_query)
            if st.session_state.useGroq:
                QA_ANSWER = st.session_state.llm(my_formatted_prompt)
            else:
                QA_ANSWER = st.session_state.llm(my_formatted_prompt)
            # PostProcess & ensure 'URLs' are formatted correctly '<URL>'    
            QA_ANSWER = wrap_urls_in_angle_brackets(QA_ANSWER)
            # PostProcess URL  -> # replace "+" in urls with "%20"
            QA_ANSWER = replace_plus_in_url_paths(QA_ANSWER)  
            # Dedup Paragraphs
            QA_ANSWER = dedup_paragraphs(QA_ANSWER)  


            # with st.expander("Short Answer"):            
            #     streamlit_add_bold_heading(st,role="assistant-0",message="Short Answer")
            #     streamlit_add_msg(st,role="assistant-0",message=QA_ANSWER)         

            # with st.expander("Summary of Docs"):        
            #     streamlit_add_bold_heading(st,role="assistant-0",message="Summary of Docs")
            #     streamlit_add_msg(st,role="assistant-0",message=WIKI_SUMMARY)

            #with st.expander("Retrieved Documents"):
            #    streamlit_add_msg(st,role="assistant-0",message=f"current_query:{user_query}") 
            #    #streamlit_add_bold_heading(st,role="assistant-0",message="Retrieved Documents")   
            #    streamlit_add_msg(st,role="assistant-0",message=MORE_DETAILS)    
#
          
            QA_ANSWER_AGGREGATOR = ''
            if configs['retrieval']['enable_aggregation']:
            ##################################################################################################################
                # Create placeholders for the intermediate responses
                col1, col2 = st.columns(2)
                with col1:
                    response1_placeholder = st.empty()
                with col2:
                    response2_placeholder = st.empty()
                # Define the height for the text areas
                text_area_height = 300  # Set the height in pixels
                # display response from LLM1
                with st.spinner("Generating response from Agent-1"):
                    # get response from the first small LLM
                    smaller_llm_1_response_output = QA_ANSWER
                    # PostProcess & ensure 'URLs' are formatted correctly '<URL>'    
                    smaller_llm_1_response_output = wrap_urls_in_angle_brackets(smaller_llm_1_response_output)
                    # PostProcess URL  -> # replace "+" in urls with "%20"
                    smaller_llm_1_response_output = replace_plus_in_url_paths(smaller_llm_1_response_output)    

                    smaller_llm_1_response_output  = nicely_format_latex_eqns(smaller_llm_1_response_output)
                    smaller_llm_1_response_output = re.sub(r'\{([^}]*)\}', r'\1', smaller_llm_1_response_output)
                    smaller_llm_1_response_output =  escape_braces(smaller_llm_1_response_output)                

                    #with st.expander('Response from Ensemble Agent-1') :                              
                    #    response1_placeholder.text_area("Response from Agent-1", smaller_llm_1_response_output, height=text_area_height)

                # display response from LLM2
                with st.spinner("Generating response from Agent-2"):
                    # get response from the first small LLM
                    if st.session_state.useGroq:
                        smaller_llm_2_response_output = st.session_state.llm_2(my_formatted_prompt)
                    else:
                        smaller_llm_2_response_output = st.session_state.llm_2(my_formatted_prompt)
                    # PostProcess & ensure 'URLs' are formatted correctly '<URL>'    
                    smaller_llm_2_response_output = wrap_urls_in_angle_brackets(smaller_llm_2_response_output)
                    # PostProcess URL  -> # replace "+" in urls with "%20"
                    smaller_llm_2_response_output = replace_plus_in_url_paths(smaller_llm_2_response_output)    

                    smaller_llm_2_response_output  = nicely_format_latex_eqns(smaller_llm_2_response_output)
                    smaller_llm_2_response_output = re.sub(r'\{([^}]*)\}', r'\1', smaller_llm_2_response_output)
                    smaller_llm_2_response_output =  escape_braces(smaller_llm_2_response_output)

                    #with st.expander('Response from Ensemble Agent-2'): 
                    #    response2_placeholder.text_area("Response from Agent-2", smaller_llm_2_response_output, height=text_area_height)

                    
                print(f'smaller_llm_1_response_output:{smaller_llm_1_response_output}')
                print(f'smaller_llm_2_response_output:{smaller_llm_2_response_output}')
                # Sanitize llm response to prevent future errors due to the presence of { } within text corpus

                aggregator_template = configs['prompts']['aggregator_template']['template']
                aggregator_template = aggregator_template.replace(
                    "smaller_llm_1_response_placeholder", smaller_llm_1_response_output)
                aggregator_template = aggregator_template.replace(
                    "smaller_llm_2_response_placeholder", smaller_llm_2_response_output)

                # print('#####################################')
                print(aggregator_template)
                # print('#####################################')
                aggregator_prompt = PromptTemplate(
                    input_variables=["history", "context", "question"],
                    template=aggregator_template,
                )
                if len(MORE_DETAILS)!=0:
                    my_formatted_aggregator_prompt = aggregator_prompt.format(history=get_chat_history_string(),context=MORE_DETAILS,question=user_query)
                    if st.session_state.useGroq:
                        QA_ANSWER_AGGREGATOR = st.session_state.llm(my_formatted_aggregator_prompt)
                    else:
                        QA_ANSWER_AGGREGATOR = st.session_state.llm(my_formatted_aggregator_prompt)
            ##################################################################################################################
    # FINAL_RESPONSE
    FINAL_RESPONSE = f'''    
    \n\n ---------------- \n\n ### **:green[Short Answer]** \n\n ---------------- \n\n  {QA_ANSWER}                         
    \n\n ---------------- \n\n ### **:green[Summary of Docs]** \n\n ---------------- \n\n  {WIKI_SUMMARY}
    '''  
    if configs['retrieval']['enable_aggregation']:
        FINAL_RESPONSE = FINAL_RESPONSE +     f'''\n\n ---------------- \n\n ### **:green[Answer Report]** \n\n ---------------- \n\n  {QA_ANSWER_AGGREGATOR}   ''' 
    return FINAL_RESPONSE, QA_ANSWER, WIKI_SUMMARY, MORE_DETAILS,filtered_retrieved_docs_LLM_answers,QA_ANSWER_AGGREGATOR


def ensembele_superposition_answer_from_docs_api_version(filtered_retrieved_docs_LLM_answers,user_query,st=None):
    MORE_DETAILS = ""
    WIKI_SUMMARY = ""
    QA_ANSWER    = ""

    # Intialize SUMMARY
    WIKI_SUMMARY = ""
    LangGraph_SUMMARY= ""
    # Intialize QA_ANSWER_AGGREGATOR
    QA_ANSWER_AGGREGATOR = ""   

    if len(filtered_retrieved_docs_LLM_answers)>0:

        # first latexify equations
        filtered_retrieved_docs_LLM_answers = latexify_documents(filtered_retrieved_docs_LLM_answers)

        # convert to string             
        MORE_DETAILS = pretty_print_docs(filtered_retrieved_docs_LLM_answers,print_output=False)

        # Use WIKI_SUMMARY
        WIKI_SUMMARY = generate_wiki_summary_from_langchain_docs_api_version(st.session_state.llm,filtered_retrieved_docs_LLM_answers,user_query)

        # LANGGRAPH SUMMARY
        # # LangGraph_SUMMARY = ""
        # with st.spinner(f"Generating LangGraph Iterative Summary of  {len(filtered_retrieved_docs_LLM_answers)} Retrieved Documents"):    
        #     # LangGraph Summary
        #     try:
        #         #LangGraph_SUMMARY = asyncio.run(run_langgraph_summarization(st.session_state.llm,filtered_retrieved_docs_LLM_answers,user_query))        
        #         breakpoint()
        #     except:   
        #            LangGraph_SUMMARY ="Error in LangGraph_SUMMARY generation. Check Code"      
        #     # PostProcess & ensure 'URLs' are formatted correctly '<URL>'    
        #     LangGraph_SUMMARY = wrap_urls_in_angle_brackets(LangGraph_SUMMARY)    
        #     LangGraph_SUMMARY = replace_plus_in_url_paths(LangGraph_SUMMARY) 
        #     streamlit_add_line(st)
        #     streamlit_add_bold_heading(st,role="assistant-0",message="Iterative Summary")        
        #     streamlit_add_msg(st,role="assistant-0",message=LangGraph_SUMMARY)          


        # Update Summary
        # WIKI_SUMMARY = WIKI_SUMMARY + "\n--------------------\n" + "\n LangGraph Summary: \n "+LangGraph_SUMMARY

        # RAPTOR_SUMMARY = ''
        ## ######################################
        ## # RAPTOR_SUMMARY
        ## ######################################
        ## # convert to list from LangChain Docs
        ## list_filtered_retrieved_docs_LLM_answers = [f"{doc.page_content} | Metadata: {doc.metadata}" for doc in filtered_retrieved_docs_LLM_answers]
        ## 
        ## with st.spinner("Recursively Summarizing Retrieved Documents (via RAPTOR)"):
        ##     RAPTOR_SUMMARY = ""
        ##     if not len(ans_list_initial) == 0:
        ##         with st.spinner("Summarizing retrived documents"):
        ##             initial_RAPTOR_SUMMARY, RAPTOR_SUMMARY_ALL_ITERATIONS = recursive_summarization(
        ##                 list_filtered_retrieved_docs_LLM_answers, user_query, faiss_embeddings, st.session_state.llm)
        ##         # Get SUMMARY
        ##         temp_summary = " \n".join(RAPTOR_SUMMARY_ALL_ITERATIONS)
        ##     
        ##         # Use RAPTOR_SUMMARY_template
        ##         my_template = configs['prompts']['RAPTOR_SUMMARY_template']['template']
        ##         my_prompt = PromptTemplate(input_variables=["summary_text","user_query"],template=my_template)
        ##         my_formatted_prompt = my_prompt.format(summary_text=temp_summary,user_query=user_query)
        ##         my_out = st.session_state.llm(my_formatted_prompt)
        ##     
        ##         temp_out = my_out
        ##     
        ##         # Regular expression to find text between <SUMMARY> and </SUMMARY>
        ##         pattern = r'<ARTICLE>(.*?)</ARTICLE>'
        ##         # Find all matches
        ##         temp_RAPTOR_SUMMARY = re.findall(pattern, temp_out, re.DOTALL)
        ##         if len(temp_RAPTOR_SUMMARY) == 0:
        ##             RAPTOR_SUMMARY = temp_out
        ##         else:
        ##             RAPTOR_SUMMARY = " \n".join(temp_RAPTOR_SUMMARY)
        ## 

        #WIKI_SUMMARY = WIKI_SUMMARY + " \n\n " + RAPTOR_SUMMARY

        # Use SUPERPOSITION_QA_ANSWER_template
        my_template = configs['prompts']['SUPERPOSITION_QA_ANSWER_template']['template']
        my_prompt = PromptTemplate(input_variables=["ans_list_final","user_query"],template=my_template)
        my_formatted_prompt = my_prompt.format(ans_list_final=MORE_DETAILS,user_query=user_query)
        if st.session_state.useGroq:
            QA_ANSWER = st.session_state.llm(my_formatted_prompt)
        else:
            QA_ANSWER = st.session_state.llm(my_formatted_prompt)
        # PostProcess & ensure 'URLs' are formatted correctly '<URL>'    
        QA_ANSWER = wrap_urls_in_angle_brackets(QA_ANSWER)
        # PostProcess URL  -> # replace "+" in urls with "%20"
        QA_ANSWER = replace_plus_in_url_paths(QA_ANSWER)  
        # Dedup Paragraphs
        QA_ANSWER = dedup_paragraphs(QA_ANSWER)  


        # with st.expander("Short Answer"):            
        #     streamlit_add_bold_heading(st,role="assistant-0",message="Short Answer")
        #     streamlit_add_msg(st,role="assistant-0",message=QA_ANSWER)         

        # with st.expander("Summary of Docs"):        
        #     streamlit_add_bold_heading(st,role="assistant-0",message="Summary of Docs")
        #     streamlit_add_msg(st,role="assistant-0",message=WIKI_SUMMARY)

        #with st.expander("Retrieved Documents"):
        #    streamlit_add_msg(st,role="assistant-0",message=f"current_query:{user_query}") 
        #    #streamlit_add_bold_heading(st,role="assistant-0",message="Retrieved Documents")   
        #    streamlit_add_msg(st,role="assistant-0",message=MORE_DETAILS)    
#
        
        QA_ANSWER_AGGREGATOR = ''

    # FINAL_RESPONSE
    FINAL_RESPONSE = f'''    
    \n\n ---------------- \n\n ### **:green[Short Answer]** \n\n ---------------- \n\n  {QA_ANSWER}                         
    \n\n ---------------- \n\n ### **:green[Summary of Docs]** \n\n ---------------- \n\n  {WIKI_SUMMARY}
    '''  
    if configs['retrieval']['enable_aggregation']:
        FINAL_RESPONSE = FINAL_RESPONSE +     f'''\n\n ---------------- \n\n ### **:green[Answer Report]** \n\n ---------------- \n\n  {QA_ANSWER_AGGREGATOR}   ''' 
    return FINAL_RESPONSE, QA_ANSWER, WIKI_SUMMARY, MORE_DETAILS,filtered_retrieved_docs_LLM_answers,QA_ANSWER_AGGREGATOR
