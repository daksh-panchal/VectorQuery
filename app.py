from flask import Flask, render_template, jsonify, request
from operator import itemgetter
from dotenv import load_dotenv
import os
from src.prompt import *
from src.helper import load_pdf_files, filter_doc, text_split, embedding_model
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

# Flask initialization.
app = Flask(__name__)

# This is the Pinecone and OpenAI API key setup.
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Loading the embedding model.
embedding = embedding_model()

index_name = "vectorquery"
# This embeds every chunk and loads the embeddings into an existing Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)

# I will create the retriever, which retrieves the most similar chunks based on their embeddings. This is done using a cosine similarity search.
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3}) # This will retrieve the top 3 most similar responses in the rank result.

# I will now connect the OpenAI LLM to generate a natural response instead of just returning the most similar chunks.
chat_model = ChatOpenAI(
    model="gpt-4o"
)

# This is the prompt template where the system prompt from prompt.py is passed to the LLM along with the user's input.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "{input}")
    ]
)

# Now I will create the entire RAG chain that will facilitate end-to-end functionality (user input to LLM response).
rag_chain = (
    {
        "context": itemgetter("input") | retriever, # This uses the retriever to conduct a similarity search and pass the most similar chunks to the LLM for context.
        "input": itemgetter("input") # The flask call sends a dict, so itemgetter extracts the input value from the dict.
    }
    | prompt # Pipe operators "|" take the output of the preceding call and use it as input for the next.
    | chat_model
    | StrOutputParser() # This converts the output of the LLM into a string for the final response.
)


# Rendering the main chat UI.
@app.route("/")
def index():
    return render_template("chat.html")

# Establishing the endpoint to handle user queries and return LLM responses.
@app.route("/chat", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response:", response)
    return response


# This runs the Flask application on localhost port 8080.

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)



