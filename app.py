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

# --- Model and Encoder Loading (Cached for performance) ---

@st.cache_resource
def load_llm_model():
    """Loads the local GPT4All model."""
    return GPT4All("orca-mini-3b-gguf2-q4_0.gguf")

@st.cache_resource
def load_encoder_model():
    """Loads the sentence transformer model for embeddings."""
    return SentenceTransformer('all-MiniLM-L6-v2')

llm_model = load_llm_model()
encoder_model = load_encoder_model()


# --- File Reading and Processing ---

def read_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def read_csv(file):
    return pd.read_csv(file).to_string()

def read_txt(file):
    stringio = StringIO(file.getvalue().decode("utf-8"))
    return stringio.read()

def get_file_content(uploaded_file):
    """Reads the content of the uploaded file based on its extension."""
    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension == ".docx": return read_docx(uploaded_file)
        if file_extension == ".pdf": return read_pdf(uploaded_file)
        if file_extension == ".csv": return read_csv(uploaded_file)
        if file_extension == ".txt": return read_txt(uploaded_file)
        st.error("Unsupported file format.")
    return None

# --- RAG Pipeline & Chat Logic ---

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    """Splits text into overlapping chunks."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]

def setup_rag_pipeline(text_content):
    """Creates embeddings and a FAISS index for the document."""
    chunks = chunk_text(text_content)
    if not chunks:
        st.warning("Could not extract text from the document.")
        return
        
    embeddings = encoder_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    st.session_state.chunks = chunks
    st.session_state.faiss_index = index

def ask_query(query, mode, model_type, api_key):
    """
    Handles the logic for both chat modes (Document RAG and General Chat).
    Returns the response and the context used (if any).
    """
    context_text = None # Initialize context as None
    
    if mode == "Chat with Document":
        if 'faiss_index' not in st.session_state:
            return "Please upload a document to use this mode.", None

        # RAG Pipeline: Retrieve relevant chunks
        query_emb = encoder_model.encode([query])
        _, I = st.session_state.faiss_index.search(np.array(query_emb), k=3)
        relevant_chunks = [st.session_state.chunks[i] for i in I[0]]
        context_text = "\n\n---\n\n".join(relevant_chunks)
        
        prompt = f"""
You are a helpful assistant. Use ONLY the following context to answer the user's question. If the answer is not found in the context, say that you don't know.

Context:
---
{context_text}
---

User Question: {query}
Answer:
"""
    else: # General Chat mode
        prompt = query # For general chat, the prompt is just the user's query

    # Generate response from the selected model
    try:
        if model_type == 'Fast Model (Gemini)':
            if not api_key:
                return "Error: Please enter your Google AI API key.", None
            genai.configure(api_key=api_key)
            gen_model = genai.GenerativeModel("gemini-2.0-flash")
            response = gen_model.generate_content(prompt)
            return response.text, context_text
        else: # Normal Model (Local)
            response = llm_model.generate(prompt, max_tokens=512)
            return response, context_text
    except Exception as e:
        return f"An error occurred: {str(e)}", None

# --- Streamlit App UI ---
st.set_page_config(page_title="Chatbot", page_icon="💬", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; }
    .stButton>button { background-color: #4285F4; color: white; border-radius: 20px; border: 1px solid #4285F4; }
    .stButton>button:hover { background-color: #FFFFFF; color: #4285F4; border: 1px solid #4285F4; }
    blockquote { background-color: #F1F3F4; border-left: 5px solid #4285F4; padding: 10px; margin: 10px 0px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'all_chats' not in st.session_state: st.session_state.all_chats = []

# --- Sidebar Controls ---
with st.sidebar:
    st.title("⚙️ Controls")

    chat_mode = st.radio("Choose Chat Mode:", ("Chat with Document", "General Chat"))

    if chat_mode == "Chat with Document":
        st.info("Upload a document to ask questions about its content.")
        uploaded_file = st.file_uploader("Upload your data file", type=["docx", "pdf", "csv", "txt"])
        
        if uploaded_file:
            if "processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name:
                with st.spinner('Reading and indexing file...'):
                    file_content = get_file_content(uploaded_file)
                    if file_content:
                        setup_rag_pipeline(file_content)
                        st.session_state.processed_file = uploaded_file.name
                        st.success("File processed successfully!")
                    else:
                        st.error("Failed to read or process the file.")
    else:
        st.info("Ask any general question. The bot will not use a document.")

    model_choice = st.selectbox("Choose a model:", ('Normal Model (Local)', 'Fast Model (Gemini)'))
    api_key = st.text_input("Enter Google AI API Key", type="password") if model_choice == 'Fast Model (Gemini)' else ""
    
    st.markdown("---")
    if st.button("New Chat"):
        st.session_state.chat_history = []
        st.rerun()

# --- Main Chat Interface ---
st.title("💬 Chatbot")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("context"):
            with st.expander("Show Sources"):
                st.markdown(f"> {message['context'].replace('---', '---')}")

# Chat input
if user_query := st.chat_input("Ask a question..."):
    # Enforce document upload for "Chat with Document" mode
    if chat_mode == "Chat with Document" and "faiss_index" not in st.session_state:
        st.warning("Please upload a document before asking questions in this mode.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, context = ask_query(user_query, chat_mode, model_choice, api_key)
                st.markdown(response)
                if context:
                    with st.expander("Show Sources"):
                        st.markdown(f"> {context.replace('---', '---')}")

                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": response, 
                    "context": context
                })

