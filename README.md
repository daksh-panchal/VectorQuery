# VectorQuery

An intelligent retrieval-augmented generation (RAG) system that combines vector embeddings with large language models to provide accurate, context-aware responses to user queries. VectorQuery leverages Pinecone for vector storage, LangChain for orchestration, and OpenAI's GPT-4 for natural language processing.

## Overview

VectorQuery is a modern AI-powered chatbot that understands documents through semantic search. Instead of keyword matching, it uses vector embeddings to find the most relevant information from the knowledge base and generates coherent responses using a large language model. This approach significantly improves answer quality and reduces hallucinations.

**Current Domain:** Thermodynamics (easily customizable to any domain)

## Key Features

- **Semantic Search:** Uses vector embeddings to find contextually relevant information, not just keyword matches.
- **Retrieval-Augmented Generation (RAG):** Combines document retrieval with LLM generation for accurate and grounded responses.
- **Production-Ready:** Built with Flask for easy deployment and scalability.
- **Context-Aware Responses:** The system retrieves the 3 most relevant document chunks to provide rich context to the LLM.
- **GPT-4 Integration:** Leverages OpenAI's GPT-4o model for high-quality natural language responses.
- **Customizable System Prompts:** Easy to adapt for different domains and use cases.
- **Web UI:** Clean & user-friendly chat interface.

## Architecture

The system is built on a modern RAG pipeline:

```
User Query
    ↓
Vector Embedding (Sentence Transformers)
    ↓
Semantic Search (Pinecone Vector Store)
    ↓
Retrieved Context (Top K Similar Chunks)
    ↓
LLM Prompt Construction
    ↓
GPT-4o Response Generation
    ↓
Formatted Response to User
```

## Tech Stack

- **Web Framework:** Flask 3.1.1
- **LLM Orchestration:** LangChain 1.2.6
- **Vector Database:** Pinecone 7.3.0
- **Embeddings:** Sentence Transformers 4.1.0
- **Language Model:** OpenAI GPT-4o
- **Deployment:** Gunicorn, Render
- **Frontend:** HTML/CSS

## Getting Started

### Prerequisites

- Python 3.8+
- Pinecone API key ([Sign up here](https://www.pinecone.io/))
- OpenAI API key ([Sign up here](https://platform.openai.com/))

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/VectorQuery.git
   cd VectorQuery
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the root directory:
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

### Usage

1. **Prepare your documents:**
   Place your PDF files in the `data/` directory

2. **Build the vector index:**
   ```bash
   python store_index.py
   ```
   This script will:
   - Load PDF files from the `data/` directory
   - Split documents into chunks
   - Generate embeddings
   - Store embeddings in Pinecone

3. **Run the application:**
   ```bash
   python app.py
   ```
   The application will start on `http://localhost:8080`

4. **Access the chat interface:**
   Open your browser and navigate to `http://localhost:8080` to start chatting

## Project Structure

```
VectorQuery/
├── app.py                 # Flask application and RAG pipeline
├── store_index.py         # Script to build vector index from documents
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup configuration
├── .env                  # Environment variables (This should contain your API keys.)
├── data/                 # PDF documents (Add your PDF files here.)
├── src/
│   ├── __init__.py
│   ├── helper.py         # PDF loading, text splitting, embeddings
│   └── prompt.py         # System prompts and instructions
├── static/
│   └── style.css         # Chat UI styling
├── templates/
│   └── chat.html         # Chat interface HTML
└── research/
    └── trials.ipynb      # Research and experimentation notebook
```

## Configuration

### Customizing the System Prompt

Edit `src/prompt.py` to change how the AI responds:

```python
system_prompt = (
    "You are a [YOUR_DOMAIN] assistant who answers user queries. "
    "Use only the retrieved context to answer the user's question..."
)
```

### Adjusting Retrieval Parameters

In `app.py`, you can modify the retriever configuration:

```python
retriever = docsearch.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 3}  # Change the k value to retrieve more/less chunks.
)
```

## How It Works

1. **Document Processing:** PDFs are loaded and split into semantic chunks.
2. **Embedding Generation:** Each chunk is converted to a vector using Sentence Transformers.
3. **Vector Storage:** Embeddings are stored in Pinecone for fast cosine similarity search.
4. **Query Processing:** User queries are embedded using the same model.
5. **Context Retrieval:** The 3 most similar chunks are retrieved from the vector store.
6. **Response Generation:** Retrieved context + user query is passed to GPT-4o.
7. **Response Refinement:** The LLM generates a concise, accurate response based on context.

## Advantages of RAG

- **Accuracy:** Responses are grounded in actual documents.
- **Reduced Hallucinations:** The LLM can only use provided context from the knowledge base.
- **Easy Updates:** Simply add new PDFs to update knowledge.
- **Cost-Effective:** Smaller context windows mean lower API costs.
- **Transparency:** Users can see which documents the AI used.

## Future Enhancements

- [ ] Support for multiple document types (Word, Excel, Web pages)
- [ ] User feedback mechanism for continuous improvement
- [ ] Chat history and session management
- [ ] Multi-user authentication
- [ ] Advanced filtering and metadata-based retrieval

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Daksh Panchal**
- Email: daksh.panchal@gmail.com
- GitHub: [@daksh-panchal](https://github.com/daksh-panchal)

## Support

For issues, questions, or suggestions, please open an [GitHub Issue](https://github.com/yourusername/VectorQuery/issues).

---

**🚀 Explore modern RAG architecture, semantic search, and intelligent document understanding with VectorQuery.**