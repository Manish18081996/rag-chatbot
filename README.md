# 📄 Resume Chatbot — RAG-Based Q&A App

This is a **Resume Chatbot** built using a **Retrieval-Augmented Generation (RAG)** pipeline. It allows users to ask questions about a candidate’s resume and receive AI-generated answers grounded in the content of the resume.

---

## 💡 What It Does

- Loads a resume (PDF format)
- Splits and embeds the content using OpenAI embeddings
- Stores chunks in a FAISS vector store
- Retrieves relevant context using semantic similarity
- Uses GPT to generate structured responses
- Presents output through a Streamlit UI

---

## 🧱 Tech Stack

- **LangChain**
- **OpenAI GPT (via `langchain-openai`)**
- **FAISS for vector storage**
- **Streamlit** for frontend
- **FastAPI** for backend

---

## 📁 Project Structure

rag-chatbot/
├── app/
│ ├── main_app.py ← Streamlit UI
│ ├── api.py ← FastAPI backend
│ ├── rag_pipeline.py ← Builds RAG chain
│ ├── data_loader.py ← Loads/splits PDF
│ ├── vector_store.py ← FAISS vector store
│ ├── prompt.yaml ← Custom prompt
├── docs/
│ └── ManishKumawatResume.pdf
├── requirements.txt
└── README.md


## ⚙️ Setup

1. **Clone the repository**

git clone https://github.com/Manish18081996/rag-chatbot.git
cd rag-chatbot
Create a virtual environment

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

Install dependencies
pip install -r requirements.txt

Set your OpenAI API Key
export OPENAI_API_KEY=your-key  # or use .env

🚀 Running the App

🖥️ Streamlit UI
streamlit run app/main_app.py
Visit http://localhost:8501 in your browser.

⚙️ FastAPI Backend

python main.py
Access API docs at: http://localhost:8000/docs

🤝 Use Cases
Recruiters screening resumes
Automated candidate insights
