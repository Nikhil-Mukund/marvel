
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader


lemmatizer = WordNetLemmatizer()

###
stop_words = set(stopwords.words('english'))

def preprocess_page_content_text(document):
    # Extract text content from the Document object
    content = document.page_content
    # Convert to lowercase
    content = content.lower()
    # Tokenize and remove stop words
    tokens = [word for word in content.split() if word not in stop_words]
    # Lemmatize
    lemmatized_text = ' '.join(
        [lemmatizer.lemmatize(token) for token in tokens])
    document.page_content = lemmatized_text
    return document


def extract_url_data(url, chunk_size=1500, chunk_overlap=100):
    print(f"extracting data from URL")
    print(f"using URL: {url}")
    loader = WebBaseLoader(url)
    data = loader.load()
    print(f"Loaded {len(data)} web documents")
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_splits = text_splitter.split_documents(data)
    print(f"Splitting web data into {len(all_splits)} chunks")
    return all_splits

from langchain.docstore.document import Document
from typing import List

from libs.retrievers import enhanced_retriever_duckduckgo


def print_langchain_documents(documents: List[Document]):
    if not documents:
        print("No documents to display.")
        return
    for i, doc in enumerate(documents, 1):
        print(f"Document {i}:")
        print("=" * 50)
        # Print relevance score
        relevance_score = doc.metadata.get('relevance_score', 'N/A')
        print(f"Relevance Score: {relevance_score:.2f}" if isinstance(
            relevance_score, float) else f"Relevance Score: {relevance_score}")
        # Print page content
        print("\nContent:")
        print(doc.page_content)
        print("=" * 50)
        print()  # Add an extra newline for spacing between documents


def diagnostic_print_RAG_chunks(st, user_input, max_return_docs):
    ####################################
    print("\n#############  BM25+ DOC SIMILARITY SEARCH RESULTS ###############")
    print("\n########################################")
    try:
        XYZ = st.session_state.bm25_retriever.get_relevant_documents(
            user_input)
        print_langchain_documents(XYZ)
        print("\n########################################")
    except:
        pass
    ####################################
    print("\n#############  FAISS DOC SIMILARITY SEARCH RESULTS ###############")
    print("\n########################################")
    try:
        XYZ = st.session_state.FAISS_retriever.get_relevant_documents(
            user_input, k=max_return_docs)
        print_langchain_documents(XYZ)
        print("\n########################################")
    except:
        pass
    ####################################
    print("\n#############  DuckDuckGo DOC SIMILARITY SEARCH RESULTS ###############")
    print("\n########################################")
    try:
        XYZ = enhanced_retriever_duckduckgo(
            st, user_input, max_return_docs=max_return_docs)
        print_langchain_documents(XYZ)
    except:
        pass
    print("\n########################################")


def ensembele_superposition_answer(st, user_query):
    print("\n ############################### \n")
    print("Executing SuperPosition Search\n")
    print("\n ############################### \n")

    # final_answer,df_MCTS_RAG = MCTS_RAG(user_query)
    # breakpoint()

    # print("##################\n");print(df_MCTS_RAG['final_answer'][0])
    with st.spinner("Fetching relevant documents"):
        docs = st.session_state.retriever.get_relevant_documents(user_query)
    with st.spinner("Verifying retrieved documents"):
        filtered_retrieved_docs = verify_and_filter_retrieved_docs(
            user_query, docs)
    ans_list_initial = filtered_retrieved_docs
    # convert ans_list_initial from langchain doc to string list
    ans_list_initial = [
        f"{doc.page_content} | Metadata: {doc.metadata}" for doc in ans_list_initial]
    # ans_list_intermediate = ' \n----------------\n '.join(ans_list_initial)

    ######################################
    # RAPTOR_SUMMARY
    ######################################
    with st.spinner("Summarizing retrived documents"):
        initial_RAPTOR_SUMMARY, RAPTOR_SUMMARY_ALL_ITERATIONS = recursive_summarization(
            ans_list_initial, user_query, faiss_embeddings, st.session_state.llm)
    # Get SUMMARY
    temp_summary = " \n".join(RAPTOR_SUMMARY_ALL_ITERATIONS)
    temp_out = st.session_state.llm("""
                                    Your task is to convert the <TEXT> into a well written technical coherent <ARTICLE> relevant to the <QUERY>.
                                    <GUIDELINES>
                                    First repeat the query within <QUERY> ... <\QUERY>.
                                    Then Carefully read the <TEXT>. 
                                    Then prepare a well-written technical coherent  <ARTICLE> from the <TEXT>. 
                                    Include all technical details, values, numbers etc.
                                    Donot show references. 
                                    <\GUIDELINES>
                                    Here is the <TEXT> {}. 
                                    Here is the <QUERY>: {}.  
                                    Generate article within  <ARTICLE> ... </ARTICLE>                                    
                                    """.format(temp_summary, user_query))
    # Regular expression to find text between <SUMMARY> and </SUMMARY>
    pattern = r'<ARTICLE>(.*?)</ARTICLE>'
    # Find all matches
    temp_RAPTOR_SUMMARY = re.findall(pattern, temp_out, re.DOTALL)
    if len(temp_RAPTOR_SUMMARY) == 0:
        RAPTOR_SUMMARY = temp_out
    else:
        RAPTOR_SUMMARY = " \n".join(temp_RAPTOR_SUMMARY)

    ######################################
    ans_list_initial_without_summary = list(ans_list_initial)
    # Append RAPTOR summary to ans_list_initial
    ans_list_initial.append(RAPTOR_SUMMARY)
    separator = "\n\n ---------------- \n\n"
    ans_list_intermediate = separator.join(ans_list_initial)
    # Optionally, you can add the separator before the first element and after the last element
    ans_list_final = f"{separator}\n\n ---- SUMMARY ------ \n\n{ans_list_intermediate}{separator}"
    # ans_list_final = "".join(ans_list_intermediate.strip().replace('nil',' '))

    ########################################

    # QA_ANSWER
    # QA_ANSWER         = st.session_state.llm(f" First Scan through the <CONTEXT> and then find the <ANSWER> to the <QUESTION> solely from the <CONTEXT>. Show corresponding source if any. Donot add any extra information or assumptions. \n <QUESTION>: {user_query} \n  <CONTEXT>: {ans_list_final} \n <ANSWER>:")
    ######################################
    # Get QA_ANSWER
    ######################################
    with st.spinner("Looking for the answers within the retrieved documents"):
        temp_out = st.session_state.llm(f'''
                                        First repeat the query within <QUERY> ... <\QUERY>.  \n
                                        Then Carefully read the <TEXT>.  \n
                                        Then carry out a needle-in-a-haystack search within the <TEXT> to find the <ANSWER> to the  <QUERY>.
                                        If there are multiple competing answers, then report discrepancies but then also pick the most relavant answer based on majority voting.
                                        Here is the <TEXT> {ans_list_final}. \n 
                                        Here is the <QUERY>: {user_query}. \n 
                                        Generate answer within  <ANSWER> ... </ANSWER>  \n
                                        ''')
    # Regular expression to find text between <ANSWER> and </ANSWER>
    pattern = r'<ANSWER>(.*?)</ANSWER>'
    # Find all matches
    QA_ANSWER = re.findall(pattern, temp_out, re.DOTALL)
    QA_ANSWER = " \n".join(QA_ANSWER)

    QA_ANSWER = temp_out
    ######################################

    MORE_DETAILS = ans_list_initial_without_summary
    # remove nil items
    MORE_DETAILS = [item for item in MORE_DETAILS if 'nil' not in item]
    MORE_DETAILS = separator.join(MORE_DETAILS)
    MORE_DETAILS = f"{separator}{MORE_DETAILS}{separator}"
    MORE_DETAILS = add_color_filename_tags(MORE_DETAILS)

    # validate_and_replace_urls
    print("\n\n Checking & validating URLS in RAPTOR_SUMMARY & MORE_DETAILS\n")
    try:
        with st.spinner("Checking & validating URLS"):
            QA_ANSWER = validate_and_replace_urls(QA_ANSWER)
            RAPTOR_SUMMARY = validate_and_replace_urls(RAPTOR_SUMMARY)
            MORE_DETAILS = validate_and_replace_urls(MORE_DETAILS)
            # WIKI_ARTICLE = validate_and_replace_urls(WIKI_ARTICLE)

            # convert to markdown supported latex
            QA_ANSWER = convert_all_latex_delimiters(QA_ANSWER)
            RAPTOR_SUMMARY = convert_all_latex_delimiters(RAPTOR_SUMMARY)
            MORE_DETAILS = convert_all_latex_delimiters(MORE_DETAILS)

    except:
        print('unable to check and validate URLs')
    # FINAL_RESPONSE
    FINAL_RESPONSE = f"\n\n ---------------- \n\n :green[QUICK ANSWER]\n\n ---------------- \n\n  {QA_ANSWER} \n\n ---------------- \n\n :green[SUMMARY OF RETRIEVED CONTENT] \n\n ---------------- \n\n {RAPTOR_SUMMARY} \n\n ---------------- \n\n :green[RETRIEVED CONTENT] \n\n ---------------- \n\n {MORE_DETAILS} \n"
    return FINAL_RESPONSE, QA_ANSWER, RAPTOR_SUMMARY, MORE_DETAILS

