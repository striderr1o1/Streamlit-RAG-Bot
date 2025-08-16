from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import streamlit as st
import ollama
from langchain_community.embeddings import OllamaEmbeddings
import os
import chromadb
from dotenv import load_dotenv
load_dotenv()
client = chromadb.Client()

def readPDF(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text+= page.extract_text()
    return text

def createEmbeddings(chunks):
    response = OllamaEmbeddings(model=os.environ.get("OLLAMA_EMBEDDING_MODEL"), base_url="http://localhost:11434").embed_documents(chunks)
    return response

def ingest_docs():
    st.header("Ingest Docs")
    uploaded_file = st.file_uploader("Upload a file", type="pdf")

    if uploaded_file is not None:
        # Read the file content
        file_content = readPDF(uploaded_file)
        st.write("File uploaded successfully")
        st.write(file_content)
        # Split the content into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = text_splitter.split_text(file_content)
        st.write("Chunks done")

        embeddings = createEmbeddings(chunks)
        st.write("Embeddings done")
        
        collection = client.get_or_create_collection(name="my_docs")
        collection.add(
            embeddings=embeddings,
            documents=chunks,
            ids=[str(i) for i in range(len(chunks))]
        )
        st.write("Documents ingested successfully")
