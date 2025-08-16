import streamlit as st
from ingestion import ingest_docs
from ingestion import createEmbeddings
import chromadb
from chromadb import Client
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()
client = chromadb.Client()

llm = ChatGroq(
    model=os.environ.get("CHAT_GROQ_MODEL"),
    api_key=os.environ.get("GROQ_API_KEY")
)
collection = client.get_or_create_collection(name="my_docs")
st.set_page_config(page_title="Streamlit Chat App", page_icon=":robot_face:", layout="wide")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["Query", "Ingestion"]
)

if page == "Ingestion":
    ingest_docs()

if page == "Query":
    st.header("Query Page")
    query = st.text_input("Enter your query:")
    if query:
        # Add your query logic here
        embeddings = createEmbeddings([query])
        results = collection.query(
            query_embeddings=embeddings,
            n_results=5  # Number of results to return
        )
        retrieved_docs = results['documents'][0]
        context= "\n".join(retrieved_docs)
        query = f"Context: {context}\n\n Query: {query}"     
        # response = llm.chat.completions.create(
        #     model="llama-3.3-70b-versatile",
        #     messages=[
        #         {"role": "user", "content": query}
        #     ]
        # )
        response = llm.invoke(query)
        st.write("Response from LLM:")
        st.write(response.content)
