import os
import nltk
import platform

from langchain_community.embeddings import OllamaEmbeddings

# Check if the NLTK packages are already downloaded
def nltk_corpora():
    try:
        if not os.path.exists(nltk.data.find('corpora/stopwords')):
            nltk.download('stopwords')
        if not os.path.exists(nltk.data.find('corpora/wordnet')):
            nltk.download('wordnet')
    except:
        print("#ERROR: NLTK data not found, make sure to go to nltk_data folder and ensure that files stopwords.zip and wordnet.zip are unzipped")


def init_environment():
    nltk_corpora()

def ragatouille_init():
    # ragatouille [& hence colbert-reranker] is not supported in windows
    system = platform.system()
    if system == "Windows":
        raise OSError('ragatouille is not supported in Windows OS')
    else:
        from ragatouille import RAGPretrainedModel
        RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        RAG.model.inference_ckpt_len_set = False
        RAG.model.config.query_maxlen = 600
        RAG.model.config.doc_maxlen = 8192
        RAG.model.config.overwrite = True
    return RAG

def faiss_embeddings_init():
    # FAISS Embeddings
    # faiss_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    faiss_embeddings = OllamaEmbeddings(
        model="nomic-embed-text", show_progress=True)  # 8192 context, SOTA
    return faiss_embeddings