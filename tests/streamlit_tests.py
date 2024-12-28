from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

model = ChatGroq(model_name="llama3-8b-8192")
messages = [SystemMessage("Translate the following sentence into French"),
            HumanMessage("Hi! How are you doing today?")]

model.invoke(messages)


