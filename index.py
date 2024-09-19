from fastapi import FastAPI
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.history import BaseChatMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    text: str
    session_id: str  

llm = ChatGroq(temperature=0.3, model_name="llama3-8b-8192")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Your name is Monica. You're an assistant that helps users. Respond to questions in 20 words or fewer and also at the end of your response, generate user message determines what emotions(Happy, Sad, Neutral, Emotional)",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

runnable = prompt | llm

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return SQLChatMessageHistory(session_id=session_id, connection="sqlite:///memory.db")

with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/chat")
async def chat(message: Message):
    """
    Endpoint to receive user messages and return chatbot responses.
    """
    input_data = {"input": message.text}

    response = with_message_history.invoke(
        input_data,
        config={"configurable": {"session_id": message.session_id}},
    )

    return {"response": response.content}
