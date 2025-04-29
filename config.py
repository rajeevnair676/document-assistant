# Config constants
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

PDF_PATH = r"C:\Users\Rajeev\Downloads\Resume_Rajeev.pdf"
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
CHUNK_SIZE = 750
CHUNK_OVERLAP = 150
DEVICE = 'cpu'
LLM_MODEL = "llama3-8b-8192"

CONTEXTUALIZE_Q_SYSTEM_PROMPT = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)
CONTEXTUALIZE_Q_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

QA_SYSTEM_PROMPT = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Keep the answer to the point."
    "{context}"
)
