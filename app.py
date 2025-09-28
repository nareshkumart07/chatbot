import streamlit as st
import time
from docx import Document
import pandas as pd
import PyPDF2
from io import StringIO
import os
import numpy as np
import faiss
from gpt4all import GPT4All
from sentence_transformers import SentenceTransformer
import google.generativeai as genai


# ------------------ Model Loading ------------------ #
@st.cache_resource
def load_llm_model():
    """Load local GPT4All model (downloads first time)."""
    return GPT4All("orca-mini-3b-gguf2-q4_0.gguf")

@st.cache_resource
def load_encoder_model():
    """Load sentence transformer for embeddings."""
    return SentenceTransformer("all-MiniLM-L6-v2")

llm_model = load_llm_model()
encoder_model = load_encoder_model()


# ------------------ File Reading ------------------ #
def read_docx(file):
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return "".join([p.extract_text() or "" for p in reader.pages])

def read_csv(file):
    return pd.read_csv(file).to_string()

def read_txt(file):
    return StringIO(file.getvalue().decode("utf-8")).read()

def get_file_content(uploaded_file):
    if uploaded_file:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext == ".docx": return read_docx(uploaded_file)
        if ext == ".pdf":  return read_pdf(uploaded_file)
        if ext == ".csv":  return read_csv(uploaded_file)
        if ext == ".txt":  return read_txt(uploaded_file)
        st.error("Unsupported format. Use .docx, .pdf, .csv or .txt.")
    return None


# ------------------ RAG Setup ------------------ #
def chunk_text(text, chunk_size=500, overlap=50):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]

def setup_rag_pipeline(text_content):
    chunks = chunk_text(text_content)
    if not chunks:
        st.warning("No text found in document.")
        return
    embeddings = encoder_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    st.session_state.chunks = chunks
    st.session_state.faiss_index = index


def ask_query(query, model_type, api_key):
    if "faiss_index" not in st.session_state or "chunks" not in st.session_state:
        return "Please upload and process a file first."

    query_emb = encoder_model.encode([query])
    _, I = st.session_state.faiss_index.search(np.array(query_emb), k=3)
    context = "\n\n".join([st.session_state.chunks[i] for i in I[0]])

    prompt = f"""
You are a helpful assistant. Use only the following context.
If the answer is not in the context, say you don't know.

Context:
---
{context}
---
User Question: {query}
Answer:
"""

    try:
        if model_type == "Fast Model (Gemini)":
            if not api_key:
                return "Error: Enter your Google AI API key in the sidebar."
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            return model.generate_content(prompt).text
        else:
            return llm_model.generate(prompt, max_tokens=512)
    except Exception as e:
        return f"Error generating response: {e}"


import streamlit as st

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("RAG-Based Chatbot")

# --- 1️⃣  Safely initialize session keys ---
if "processed_file" not in st.session_state:
    st.session_state["processed_file"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# --- 2️⃣  File uploader ---
uploaded_file = st.file_uploader(
    "Upload your document",
    type=["pdf", "docx", "txt", "csv"]
)

# --- 3️⃣  Process new file only if changed ---
if uploaded_file:
    # Compare safely with stored name
    if st.session_state.get("processed_file") != uploaded_file.name:
        with st.spinner("Processing file..."):
            # ↓↓↓ Replace this with your own document ingestion logic ↓↓↓
            file_content = uploaded_file.read().decode(errors="ignore")
            # (e.g., create embeddings, build vector store, etc.)
            # ↑↑↑ End of your processing logic ↑↑↑
        st.session_state["processed_file"] = uploaded_file.name
        st.success(f"✅ File '{uploaded_file.name}' processed successfully!")
    else:
        st.info("This file is already processed.")

# --- 4️⃣  Chat interface ---
user_query = st.text_input("Ask a question about your document:")

if st.button("Send") and user_query:
    with st.spinner("Generating answer..."):
        # ↓↓↓ Replace this with your Gemini/GPT4All RAG logic ↓↓↓
        answer = f"Dummy answer to: {user_query}"
        # ↑↑↑ End of your RAG logic ↑↑↑
    st.session_state.chat_history.append((user_query, answer))

# --- 5️⃣  Display chat history ---
if st.session_state.chat_history:
    st.subheader("Chat History")
    for q, a in st.session_state.chat_history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
        st.markdown("---")



