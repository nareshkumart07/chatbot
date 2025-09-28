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


# ------------------ Streamlit UI ------------------ #
st.set_page_config(page_title="Chat with your Data", page_icon="💬", layout="wide")
st.markdown("""
<style>
.stApp {background-color: #F0F8FF;}
.stButton>button {
    background-color: #4F8BF9; color: white; border-radius: 20px; border: 1px solid #4F8BF9;
}
.stButton>button:hover {
    background-color: #FFFFFF; color: #4F8BF9; border: 1px solid #4F8BF9;
}
</style>
""", unsafe_allow_html=True)

# Initialise session state
for key, val in [
    ("chat_history", []),
    ("last_fast_model_time", 0),
    ("fast_model_count", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = val


# ------------------ Sidebar ------------------ #
with st.sidebar:
    st.title("📄 Chat with your Data")
    uploaded_file = st.file_uploader(
        "Upload file", type=["docx", "pdf", "csv", "txt"], key="file_uploader"
    )

    # ✅ Safe check for processed_file
    if uploaded_file and st.session_state.get("processed_file") != uploaded_file.name:
        with st.spinner("Reading and indexing file..."):
            text_content = get_file_content(uploaded_file)
            if text_content:
                setup_rag_pipeline(text_content)
                st.session_state.processed_file = uploaded_file.name
                st.success("File processed successfully!")
            else:
                st.error("Failed to read or process the file.")

    model_choice = st.selectbox(
        "Choose a model:", ("Normal Model (Local)", "Fast Model (Gemini)")
    )
    api_key = (
        st.text_input("Google AI API Key", type="password")
        if model_choice == "Fast Model (Gemini)"
        else ""
    )

    st.markdown("""
---
**Model Info**
- **Normal Model:** Runs locally (private, slower).
- **Fast Model:** Uses Gemini API (faster, needs API key, 2 Qs/min).
""")


# ------------------ Main Chat ------------------ #
st.title("💬 Chatbot")

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_query := st.chat_input("Ask a question about your document..."):
    if "faiss_index" not in st.session_state:
        st.warning("Please upload a document before asking questions.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Rate limit for fast model
        blocked = False
        if model_choice == "Fast Model (Gemini)":
            now = time.time()
            if now - st.session_state.last_fast_model_time < 60:
                if st.session_state.fast_model_count >= 2:
                    wait = 60 - (now - st.session_state.last_fast_model_time)
                    st.warning(f"Fast model limit reached. Wait {int(wait)} seconds.")
                    blocked = True
                else:
                    st.session_state.fast_model_count += 1
            else:
                st.session_state.last_fast_model_time = now
                st.session_state.fast_model_count = 1

        if not blocked:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = ask_query(user_query, model_choice, api_key)
                    st.markdown(answer)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )
