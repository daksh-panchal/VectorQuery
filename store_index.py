from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_doc, text_split, embedding_model
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

extracted_data = load_pdf_files("data")
minimal_doc = filter_doc(extracted_data)
chunk_text = text_split(minimal_doc)
embedding = embedding_model()

pinecone_api_key = PINECONE_API_KEY
pinecone_client = Pinecone(api_key=pinecone_api_key) # This authenticates with my pinecone account.


# Now, I will make an index in Pinecone so that I can begin to store vector embeddings.
index_name = "vectorquery"

if not pinecone_client.has_index(index_name):
    pinecone_client.create_index(
        name = index_name,
        dimension=384, # The vector length as stated on the HuggingFace website and also verified with the test query vector.
        metric="cosine", # Here I am using the cosine similarity search, which looks at directional similarity of embeddings.
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pinecone_client.Index(index_name)

# The index is created in Pinecone so now I will store the vector embeddings for all the chunks.
docsearch = PineconeVectorStore.from_documents(
    documents = chunk_text,
    embedding = embedding, #running the embedding_model() function which contains the actual model from HuggingFace
    index_name = index_name
)


