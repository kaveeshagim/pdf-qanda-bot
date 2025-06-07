import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Setup
st.set_page_config(page_title="PDF Q&A Bot")
st.title("📄 Ask your PDF anything!")

# Input OpenAI API Key
openai_api_key = st.text_input("🔑 Enter OpenAI API Key", type="password")
uploaded_file = st.file_uploader("📁 Upload a PDF", type=["pdf"])

if uploaded_file and openai_api_key:
    with st.spinner("Processing PDF..."):
        # Save uploaded file to disk
        file_path = f"./{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Load PDF and split into pages
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        # Embed pages and store in vector DB
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        db = FAISS.from_documents(pages, embeddings)

        # Setup LLM + Retriever Chain
        llm = ChatOpenAI(openai_api_key=openai_api_key)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

        st.success("PDF processed! Ask away 👇")
        query = st.text_input("❓ Ask a question")

        if query:
            with st.spinner("Thinking..."):
                answer = qa_chain.run(query)
                st.markdown(f"**Answer:** {answer}")
