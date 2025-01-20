from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader,PyMuPDFLoader,PDFPlumberLoader
from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_groq import ChatGroq
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory.buffer import ConversationBufferMemory
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
    ConversationalRetrievalChain,
    )
from langchain.chains.combine_documents import create_stuff_documents_chain
from tqdm import tqdm
import faiss
from uuid import uuid4
import config
import warnings

warnings.filterwarnings('ignore')


def pdf_parser(pdf_path):
    print("Reading the PDF file....")
    loader = PyMuPDFLoader(pdf_path)
    doc = loader.load()
    return doc

def word_parser():
    pass

def doc_text_splitter(document,chunk_size,chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                 chunk_overlap=chunk_overlap)
    print("Splitting the text....")
    doc_split = splitter.split_documents(document)
    return doc_split

def create_embeddings(model,device='cpu'):
    model_kwargs = {'device':device}
    embeddings = HuggingFaceEmbeddings(model_name=model,
                           model_kwargs=model_kwargs)
    # embed_docs = embeddings.embed_documents([doc.page_content for doc in tqdm(split_doc)])
    return embeddings

def create_vector_store(embeddings,documents):
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    vector_store = FAISS(embedding_function=embeddings,
                     index=index,
                     docstore=InMemoryDocstore(),
                     index_to_docstore_id={})
    uuids = [str(uuid4()) for _ in range(len(documents))]
    print("Adding document to the vector store....")
    vector_store.add_documents(documents=documents, ids=uuids)
    return vector_store

def create_retriever(vector_store,llm,contextualize_q_prompt,qa_system_prompt):
    retriever = vector_store.as_retriever()
    print("Getting retrievers and RAG chains ready....")
    rag_retriever = create_history_aware_retriever(
                llm,retriever, contextualize_q_prompt)
    qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(llm,qa_prompt)
    return rag_retriever,qa_chain

def chat(query,rag_retriever,qa_chain):
    memory = ConversationBufferMemory(memory_key='chat_history',
                                  return_messages=True)
    chat_history = memory.load_memory_variables({})['chat_history']
    rag_chain = create_retrieval_chain(rag_retriever, qa_chain)
    response = rag_chain.invoke({
        "chat_history":chat_history,
        "input":query})
    

    memory.save_context(
        {"input": query},
        {"output": response["answer"]}
    )

    return response["answer"]


if __name__ == "__main__":
    load_dotenv()
    document = pdf_parser(config.PDF_PATH)
    doc_split = doc_text_splitter(document=document,
                                  chunk_size=config.CHUNK_SIZE,
                                  chunk_overlap=config.CHUNK_OVERLAP)
    embeddings = create_embeddings(model=config.MODEL_NAME,
                                    device=config.DEVICE
    )
    vector_store = create_vector_store(embeddings,doc_split)
    llm = ChatGroq(model_name=config.LLM_MODEL)
    rag_retriever,qa_chain = create_retriever(vector_store,
                                              llm,
                                              config.CONTEXTUALIZE_Q_PROMPT,
                                              config.QA_SYSTEM_PROMPT
    )

    query = str(input("Me: "))
    while query.casefold()!="end":
        print(query.casefold())
        response = chat(query,rag_retriever,qa_chain)
        print("Assistant: ",response,"\n")
        query = str(input("Enter your query: "))