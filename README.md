# Streamlit RAG Chatbot

A simple Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, LangChain, Ollama, and ChromaDB.

## Features
- **Document Ingestion:** Upload PDF files, split into chunks, and generate embeddings.
- **Vector Store:** Store document embeddings in ChromaDB for efficient retrieval.
- **Chat Interface:** Query your documents and get AI-powered answers using an LLM.
- **Sidebar Navigation:** Easily switch between ingestion and query pages.

## Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set environment variables
Set your embedding model name for Ollama:
```bash
export OLLAMA_EMBEDDING_MODEL="nomic-embed-text"
export GROQ_API_KEY="xyz"
export CHAT_GROQ_MODEL="llama-3.3-70b-versatile"
```

### 3. Run the app
```bash
streamlit run main.py
```

## Usage
- Go to the **Ingestion** page to upload and process your documents.
- Switch to the **Query** page to ask questions about your documents.

## Project Structure
```
main.py           # Main Streamlit app with sidebar navigation, ask queries
ingestion.py      # Document ingestion logic      
README.md         # Project documentation
pyproject.toml    # Python project metadata
```

## Note
- Project is not production ready. Issues may arise in dependencies. If any issue arises, do not hesitate to email me at: mnipk1243@gmail.com

## License
MIT
