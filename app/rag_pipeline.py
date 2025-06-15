# RAG setup: Retrieval + Response generation

from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableMap
from langchain_core.output_parsers import PydanticOutputParser  
from langchain_core.runnables import RunnableLambda
from app.llm_chain import get_llm
import yaml
from pydantic import BaseModel, Field

#Function of Loading prompt from YAML file.
def load_prompt():
    with open("app/prompt.yaml", "r") as prompt_file:
        data = yaml.safe_load(prompt_file)
        return data["prompt"]
    
# Pydantic class for strucured output    
class QAResponse(BaseModel):
    answer: str = Field(..., description="Concise answer to the user's question based on the resume context.")
    confidence_level: str = Field(..., description="Confidence level of the answer. One of: High, Medium, Low.")


# load prompt template
prompt_template = load_prompt()

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = get_llm()

    # Using pydantic output parser
    parser = PydanticOutputParser(pydantic_object=QAResponse)

    # Create a document chain that uses prompt → LLM → parser
    document_chain = QA_PROMPT | llm | parser

    # Combine retriever and document chain
    rag_chain = RunnableMap({
    "context": RunnableLambda(lambda x: x["question"]) | retriever,
    "question": lambda x: x["question"]
    }) | document_chain

    return rag_chain
