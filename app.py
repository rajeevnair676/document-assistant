import streamlit as st
from typing import List
from dotenv import load_dotenv
from docx import Document
import pdfplumber
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.schema import Document as LangchainDocument
from langchain_text_splitters.spacy import SpacyTextSplitter
from pathlib import Path
import os
import pickle
import time

# Set up Streamlit configuration
st.set_page_config(page_icon="📄", layout="wide", page_title="Doc & PDF Chatbot with Context")

def icon(emoji: str):
    """Display an emoji as a page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )

icon("🤖")
st.subheader("Document Assistant", divider="rainbow", anchor=False)

load_dotenv()

VECTORSTORE_PATH = "./vectorstore.pkl"

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

def parse_word_document(uploaded_file) -> List[LangchainDocument]:
    """Parse a Word document into LangChain Document objects."""
    doc = Document(uploaded_file)
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    return split_document(text)

def parse_pdf_document(uploaded_file) -> List[LangchainDocument]:
    """Parse a PDF document into LangChain Document objects."""
    text = ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return split_document(text)
    except Exception as e:
        st.error(f"Failed to process the PDF document: {e}", icon="🚨")
        return []

def split_document(text, max_chunk_size=500) -> List[LangchainDocument]:
    """Split text into manageable chunks."""
    words = text.split()
    chunks = [" ".join(words[i:i + max_chunk_size]) for i in range(0, len(words), max_chunk_size)]
    return [LangchainDocument(page_content=chunk) for chunk in chunks]

def load_vectorstore():
    """Load the vectorstore from disk if it exists."""
    if os.path.exists(VECTORSTORE_PATH):
        with open(VECTORSTORE_PATH, "rb") as f:
            st.session_state.vectorstore = pickle.load(f)
            st.success("Loaded the previous vectorstore from disk.", icon="✅")
    else:
        st.session_state.vectorstore = None

def save_vectorstore(vectorstore):
    """Save the vectorstore to disk."""
    with open(VECTORSTORE_PATH, "wb") as f:
        pickle.dump(vectorstore, f)
        st.success("Vectorstore saved to disk.", icon="✅")

def initialize_vectorstore(documents: List[LangchainDocument]):
    """Initialize a FAISS vectorstore with precomputed embeddings."""
    if st.session_state.vectorstore is None:
        with st.spinner("Generating embeddings and initializing vectorstore. This may take a moment..."):
            time.sleep(0.5)  # Simulate loading time
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            vectorstore = FAISS.from_documents(documents, embeddings)
            st.session_state.vectorstore = vectorstore
            save_vectorstore(vectorstore)

def initialize_qa_chain():
    """Initialize a conversational QA chain using the vectorstore."""
    if st.session_state.vectorstore:
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 2})
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
            ChatGroq(temperature=0, model_name="llama3-8b-8192"),
            retriever=retriever,
            memory=memory,
        )
        st.success("QA chain initialized.", icon="✅")

# Load existing vectorstore if available
load_vectorstore()

# File uploader for Word and PDF documents
uploaded_file = st.sidebar.file_uploader("Upload a Word or PDF document (.docx, .pdf)", type=["docx", "pdf"])

if uploaded_file:
    if os.path.exists(VECTORSTORE_PATH):
        os.remove(VECTORSTORE_PATH)
        st.session_state.vectorstore = None
        st.success("Previous vectorstore deleted due to new document upload.", icon="🗑️")

    with st.spinner("Processing uploaded document..."):
        time.sleep(0.5)  # Simulate loading time
        if uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            documents = parse_word_document(uploaded_file)
        elif uploaded_file.type == "application/pdf":
            documents = parse_pdf_document(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a .docx or .pdf file.", icon="🚨")
            documents = []

    if documents:
        with st.spinner("Initializing vectorstore and QA chain..."):
            time.sleep(0.5)  # Simulate processing time
            initialize_vectorstore(documents)
            initialize_qa_chain()

if prompt := st.chat_input("Enter your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        with st.spinner("Generating response..."):
            time.sleep(0.5)  # Simulate processing time
            if st.session_state.qa_chain:
                response = st.session_state.qa_chain({"question": prompt, "chat_history": st.session_state.messages})
                assistant_response = response.get("answer", "I'm sorry, I couldn't find an answer.")
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                for message in st.session_state.messages:
                    role = "user" if message["role"] == "user" else "assistant"
                    avatar = "👨‍💻" if role == "user" else "🤖"
                    with st.chat_message(role, avatar=avatar):
                        st.markdown(message["content"])
            else:
                st.warning("Please upload a document to initialize the QA system.")
    except Exception as e:
        st.error(f"Error during QA processing: {e}", icon="🚨")