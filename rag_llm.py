from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.6
)

def llm_chat(prompt: str) -> str:
    return llm.invoke(prompt).content

def llm_chat_stream(prompt: str):
    for chunk in llm.stream(prompt):
        if chunk.content:
            yield chunk.content
