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
    # This will download the model on the first run if not already present
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
        if file_extension == ".docx":
            return read_docx(uploaded_file)
        elif file_extension == ".pdf":
            return read_pdf(uploaded_file)
        elif file_extension == ".csv":
            return read_csv(uploaded_file)
        elif file_extension == ".txt":
            return read_txt(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a .docx, .pdf, .csv, or .txt file.")
            return None
    return None

# --- RAG Pipeline Functions ---

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    """Splits text into overlapping chunks."""
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def setup_rag_pipeline(text_content):
    """Creates embeddings and a FAISS index for the document."""
    chunks = chunk_text(text_content)
    if not chunks:
        st.warning("Could not extract text from the document.")
        return None, None
        
    embeddings = encoder_model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    
    st.session_state.chunks = chunks
    st.session_state.faiss_index = index
    return chunks, index

def ask_query(query, model_type, api_key):
    """
    Performs RAG to answer a query.
    """
    if 'faiss_index' not in st.session_state or 'chunks' not in st.session_state:
        return "Please upload and process a data file first."

    # 1. Retrieve relevant chunks
    query_emb = encoder_model.encode([query])
    k = 3  # number of relevant chunks
    _, I = st.session_state.faiss_index.search(np.array(query_emb), k)
    relevant_chunks = [st.session_state.chunks[i] for i in I[0]]
    context_text = "\n\n".join(relevant_chunks)

    # 2. Create the prompt
    prompt = f"""
You are a helpful assistant. Use only the following context to answer the user's question. If the answer is not in the context, say that you don't know.

Context:
---
{context_text}
---

User Question: {query}
Answer:
"""
    # 3. Generate response from the selected model
    try:
        if model_type == 'Fast Model (Gemini)':
            if not api_key:
                return "Error: Please enter your Google AI API key in the sidebar to use the Fast Model."
            genai.configure(api_key=api_key)
            gen_model = genai.GenerativeModel("gemini-1.5-flash-latest")
            response = gen_model.generate_content(prompt)
            return response.text
        else: # Normal Model (Local)
            response = llm_model.generate(prompt, max_tokens=512)
            return response
    except Exception as e:
        return f"An error occurred: {str(e)}"

# --- Streamlit App UI ---

st.set_page_config(page_title="Chat with your Data", page_icon="💬", layout="wide")

# Custom CSS for the blue and white theme
st.markdown("""
<style>
    .stApp {
        background-color: #F0F8FF;
    }
    .stButton>button {
        background-color: #4F8BF9; color: white; border-radius: 20px; border: 1px solid #4F8BF9;
    }
    .stButton>button:hover {
        background-color: #FFFFFF; color: #4F8BF9; border: 1px solid #4F8BF9;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_fast_model_time' not in st.session_state:
    st.session_state.last_fast_model_time = 0
if 'fast_model_count' not in st.session_state:
    st.session_state.fast_model_count = 0

# --- Sidebar Controls ---
with st.sidebar:
    st.title("📄 Chat with your Data")
    st.markdown("Upload a document and ask questions about its content.")

    uploaded_file = st.file_uploader("Upload your data file", type=["docx", "pdf", "csv", "txt"], key="file_uploader")
    
    if uploaded_file and "processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name:
        with st.spinner('Reading and indexing file... This may take a moment.'):
            file_content = get_file_content(uploaded_file)
            if file_content:
                setup_rag_pipeline(file_content)
                st.session_state.processed_file = uploaded_file.name
                st.success("File processed successfully!")
            else:
                st.error("Failed to read or process the file.")


    model_choice = st.selectbox("Choose a model:", ('Normal Model (Local)', 'Fast Model (Gemini)'))
    
    api_key = ""
    if model_choice == 'Fast Model (Gemini)':
        api_key = st.text_input("Enter your Google AI API Key", type="password")

    st.markdown("""
    ---
    **Model Information:**
    - **Normal Model:** Runs locally on your computer. Slower, but private and no limits.
    - **Fast Model:** Uses Google's Gemini API. Faster, but requires an API key and has a rate limit (2 Qs/min).
    """)


# --- Main Chat Interface ---
st.title("💬 Chatbot")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if user_query := st.chat_input("Ask a question about your document..."):
    if "faiss_index" not in st.session_state:
        st.warning("Please upload a document before asking questions.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Rate limiting for fast model
        is_rate_limited = False
        if model_choice == 'Fast Model (Gemini)':
            current_time = time.time()
            if current_time - st.session_state.last_fast_model_time < 60:
                if st.session_state.fast_model_count >= 2:
                    is_rate_limited = True
                    wait_time = 60 - (current_time - st.session_state.last_fast_model_time)
                    st.warning(f"Fast model limit reached. Please wait {int(wait_time)} seconds.")
                else:
                    st.session_state.fast_model_count += 1
            else:
                st.session_state.last_fast_model_time = current_time
                st.session_state.fast_model_count = 1

        if not is_rate_limited:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = ask_query(user_query, model_choice, api_key)
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})

