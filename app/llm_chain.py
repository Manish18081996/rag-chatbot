#Langchain pipeline setup with OpenAI LLM

from langchain.chat_models import ChatOpenAI
from app.config import OPENAI_API_KEY, LLM_MODEL_NAME

def get_llm():
    return ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name=LLM_MODEL_NAME,
        temperature=0.2
    )