# Vector store setup (FAISS)

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from app.config import OPENAI_API_KEY

def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    return vectorstore