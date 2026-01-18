from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document

# Extracting the text from PDF files
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents

# With a list of document objects, this function will return a new list of document objects containing only the source and page_content data.
def filter_doc(docs: List[Document]) -> List[Document]:
    minimal_doc: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_doc.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    
    return minimal_doc

# This function will perform chunking; splitting the text content into smaller chunks.
def text_split(minimal_doc):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 20
    )

    chunk_text = text_splitter.split_documents(minimal_doc)
    return chunk_text

# Now, the chunks must be passed into an embedding model so that each chunk has a corresponding vector embedding.
# First, I will download and return the HuggingFace embedding model so it can be used to create the embeddings.

from langchain_huggingface import HuggingFaceEmbeddings

def embedding_model():
    model_name ="sentence-transformers/all-MiniLM-L6-v2" # This model name was obtained from the HuggingFace website.
    embeddings = HuggingFaceEmbeddings(
        model_name = model_name # Setting the model_name value in the HuggingFaceEmbeddings function to the model_name that I obtained above.
    )
    return embeddings

embedding = embedding_model() # Calling the function which downloads and returns the embedding model.