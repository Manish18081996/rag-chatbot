# Streamlit-based frontend
import sys
import os
import streamlit as st
import json
from pydantic import ValidationError

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader import load_and_split_documents
from vector_store import create_vector_store
from rag_pipeline import build_rag_chain

st.set_page_config(page_title="Resume Chatbot")
st.title("💬 Ask About Candidate Resume")

# Load resume and build the chain
@st.cache_resource
def setup_rag_chain():
    chunks = load_and_split_documents("docs/ManishKumawatResume.pdf")
    vectorstore = create_vector_store(chunks)
    return build_rag_chain(vectorstore)

rag_chain = setup_rag_chain()

# Input and output UI
question = st.text_input("Ask a question about the candidate:")

if question:
    with st.spinner("Thinking..."):
        try:
            # Call the chain
            response_model = rag_chain.invoke({"question": question})
            
            # If response is a pydantic object
            if hasattr(response_model, 'dict'):
                response = response_model.dict()
            else:
                response = json.loads(response_model)

            # Nicely formatted output
            st.markdown(f"""
                <div style="border: 1px solid #ccc; border-radius: 10px; padding: 15px; background-color: #f9f9f9;">
                    <h4>🧠 Answer</h4>
                    <p>{response.get("answer", "No answer provided.")}</p>
                    <h4>📈 Confidence Level</h4>
                    <p>{response.get("confidence_level", "Not specified.")}</p>
                </div>
            """, unsafe_allow_html=True)

        except (ValidationError, json.JSONDecodeError, AttributeError, TypeError) as e:
            st.error(f"❌ Failed to parse structured response: {e}")
            st.markdown("### Raw response:")
            st.code(str(response_model), language="json")
