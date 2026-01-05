

# Save FAISS vectorstore
import os
import pickle
import uuid
from langchain.schema import Document


def save_faiss_vectorstore(faiss_vectorstore, directory_name, filename="FAISS_PICKLE"):
    print("saving faiss vectorstore...")
    # Create the directory if it doesn't exist
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    filepath = os.path.join(directory_name, filename+".pkl")
    with open(filepath, "wb") as f:
        pickle.dump(faiss_vectorstore, f)
        print("Saved faiss vectorstore.")


# load FAISS vectorStore
def load_faiss_vectorstore(faiss_directory, filename="FAISS_PICKLE"):
    # Load the .pkl file
    filepath = os.path.join(faiss_directory, filename+".pkl")
    with open(filepath, "rb") as f:
        faiss_vectorstore = pickle.load(f)
    return faiss_vectorstore


def get_docs_from_faiss_vectorstore_old(faiss_vectorstore):
    vectorstore_docs = []
    if isinstance(faiss_vectorstore, dict):
        print("The faiss_vectorstore PICKLE file is a dictionary")
        vectorstore_docs = [
            doc for doc in faiss_vectorstore['vectorstore_docs']]
    else:
        print("The faiss_vectorstore PICKLE file is a langchain_community.vectorstores.faiss.FAISS")
        total_docs = len(faiss_vectorstore.index_to_docstore_id)
        for i, docstore_id in enumerate(faiss_vectorstore.index_to_docstore_id.values()):
            try:
                doc = faiss_vectorstore.docstore.search(docstore_id)
                vectorstore_docs.append(doc)
            except KeyError:
                print(
                    f"Warning: Document with ID {docstore_id} not found in docstore.")
            # Print progress every 1000 documents
            # if (i + 1) % 1000 == 0 or (i + 1) == total_docs:
                # print(f"Processed {i + 1}/{total_docs} documents")
    print(f"\nTotal documents retrieved: {len(vectorstore_docs)}")
    return vectorstore_docs



def get_docs_from_faiss_vectorstore(faiss_vectorstore):
    """
    Retrieves documents from a FAISS vectorstore (a langchain_community.vectorstores.faiss.FAISS
    object), assigns each a unique UUID 'id', and returns the new list of Documents.
    """
    vectorstore_docs = []
    # This is specifically for the community FAISS vectorstore.
    # 'faiss_vectorstore.index_to_docstore_id' contains a mapping from internal index to docstore ID.
    total_docs = len(faiss_vectorstore.index_to_docstore_id)
    for i, docstore_id in enumerate(faiss_vectorstore.index_to_docstore_id.values()):
        try:
            # old_doc should be a Document-like object (without an 'id' attribute)
            old_doc = faiss_vectorstore.docstore.search(docstore_id)

            # Create a new Document with a UUID
            new_doc = Document(
                page_content=old_doc.page_content,
                metadata=old_doc.metadata,
                id=str(uuid.uuid4())  # Generate a unique ID
            )
            vectorstore_docs.append(new_doc)
        except KeyError:
            print(f"Warning: Document with ID {docstore_id} not found in docstore.")
        # Optional progress print
        # if (i + 1) % 1000 == 0 or (i + 1) == total_docs:
        #     print(f"Processed {i + 1}/{total_docs} documents")
    print(f"\nTotal documents retrieved: {len(vectorstore_docs)}")
    return vectorstore_docs
