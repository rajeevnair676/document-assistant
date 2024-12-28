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
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.schema import Document as LangchainDocument
from langchain_text_splitters.spacy import SpacyTextSplitter
from pathlib import Path

# import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using {device}..")

st.set_page_config(page_icon="📄", layout="wide", page_title="Doc & PDF Chatbot with Context")

def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )

icon("🤖")

st.subheader("Document Assistant", divider="rainbow", anchor=False)

load_dotenv()

# Initialize chat history and QA chain
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def pdf_loader(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    document = loader.load()
    return document

def parse_lc_pdf(pdf_path):
    # pdf_path = Path(pdf_path)
    print(pdf_path)
    temp_file = "./temp.pdf"
    with open(temp_file, "wb") as file:
       file.write(uploaded_file.getvalue())
       file_name = uploaded_file.name
    document = pdf_loader(temp_file)
    text_splitter = SpacyTextSplitter(pipeline='en_core_web_sm',max_length=10000)
    doc_split = text_splitter.split_documents(document)
    return [LangchainDocument(chunk.page_content) for chunk in doc_split]


def parse_word_document(uploaded_file) -> List[LangchainDocument]:
    """Parse the uploaded Word document into LangChain Document objects."""
    doc = Document(uploaded_file)
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    chunks = split_document(text)
    return [LangchainDocument(page_content=chunk) for chunk in chunks]

def parse_pdf_document(uploaded_file) -> List[LangchainDocument]:
    """Parse the uploaded PDF document into LangChain Document objects using pdfplumber."""
    text = ""
    try:
        # Open the PDF file
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""  # Extract text from each page
        
        # Split the text into manageable chunks
        chunks = split_document(text)
        return [LangchainDocument(page_content=chunk) for chunk in chunks]
    except Exception as e:
        st.error(f"Failed to process the PDF document: {e}", icon="🚨")
        return []

def split_document(text, max_chunk_size=500):
    """Split the document into smaller chunks to fit token limits."""
    words = text.split()
    for i in range(0, len(words), max_chunk_size):
        yield " ".join(words[i:i + max_chunk_size])

def initialize_chain(documents: List[LangchainDocument]):
    """Initialize a conversational QA chain with uploaded documents."""
    model_kwargs = {'device': 'cuda'}
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",model_kwargs=model_kwargs)
    print("[INFO] Embeddings generated...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    print("[INFO] FAISS embeddings computed...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # Retrieve top 5 relevant chunks
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        ChatGroq(temperature=0, model_name="llama3-8b-8192"),
        retriever=retriever,
        memory=memory,
    )
    return qa_chain

# File uploader for Word and PDF documents
uploaded_file = st.file_uploader("Upload a Word or PDF document (.docx, .pdf)", type=["docx", "pdf"],
                                 )
# print(f"[FILE]: {uploaded_file.filename}")

if uploaded_file:
    if uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        documents = parse_word_document(uploaded_file)
    elif uploaded_file.type == "application/pdf":
        documents = parse_lc_pdf(uploaded_file)
    else:
        st.error("Unsupported file type. Please upload a .docx or .pdf file.", icon="🚨")
        documents = []
    print("[INFO] Documents parsed...")
    if documents:
        st.session_state.qa_chain = initialize_chain(documents)
        st.success("Document processed and QA chain initialized!", icon="✅")

if prompt := st.chat_input("Enter your question here..."):
    # Truncate prompt if necessary
    MAX_PROMPT_LENGTH = 512
    if len(prompt.split()) > MAX_PROMPT_LENGTH:
        prompt = " ".join(prompt.split()[:MAX_PROMPT_LENGTH])

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar='👨‍💻'):
        st.markdown(prompt)

    # Generate response
    try:
        if st.session_state.qa_chain:
            response = st.session_state.qa_chain({"question": prompt, "chat_history": st.session_state.chat_history})
            assistant_response = response.get("answer", "I'm sorry, I couldn't find an answer.")
            st.session_state.chat_history.append({"question": prompt, "answer": assistant_response})
        else:
            assistant_response = "Please upload a document to initialize the QA system."

        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(assistant_response)

        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    except Exception as e:
        st.error(f"Error during QA processing: {e}", icon="🚨")