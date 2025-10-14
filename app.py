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
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# --- Model and Encoder Loading (Cached for performance) ---

@st.cache_resource
def load_llm_model():
    """Loads the 'Normal' local GPT4All model."""
    return GPT4All("orca-mini-3b-gguf2-q4_0.gguf")

@st.cache_resource
def load_encoder_model():
    """Loads the sentence transformer model for embeddings."""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_advanced_llm_model():
    """Loads the 'Advanced' quantized DeepSeek model from Hugging Face."""
    model_name = "deepseek-ai/DeepSeek-R1-0528"
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16 # Ensure compatibility
    )
    return model, tokenizer

# Load all models at startup
llm_model = load_llm_model()
encoder_model = load_encoder_model()
advanced_model, advanced_tokenizer = load_advanced_llm_model()


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

def ask_query(query, model_type, api_key, chat_history):
    """
    Performs RAG to answer a query, now including conversation history for context.
    Returns the response and the document context used.
    """
    if 'faiss_index' not in st.session_state:
        return "Please upload and process a file first.", None

    query_emb = encoder_model.encode([query])
    _, I = st.session_state.faiss_index.search(np.array(query_emb), k=3)
    relevant_chunks = [st.session_state.chunks[i] for i in I[0]]
    doc_context = "\n\n---\n\n".join(relevant_chunks)

    history_str = ""
    recent_history = chat_history[-4:]
    if recent_history:
        history_str += "Here is the recent conversation history:\n"
        for msg in recent_history:
            history_str += f"{msg['role'].capitalize()}: {msg['content']}\n"

    prompt = f"""
You are a helpful AI assistant. Answer the user's latest question based on the conversation history and the provided document context.

{history_str}

Use the following context from the document if it's relevant. If the answer isn't in the document or history, say you don't know.

Document Context:
---
{doc_context}
---

User's Latest Question: {query}
Answer:
"""
    try:
        if model_type == 'Fast Model (Gemini)':
            if not api_key:
                return "Error: Please enter your Google AI API key.", None
            genai.configure(api_key=api_key)
            gen_model = genai.GenerativeModel("gemini-2.0-flash")
            response = gen_model.generate_content(prompt)
            return response.text, doc_context
            
        elif model_type == 'Advanced Model (DeepSeek)':
            inputs = advanced_tokenizer(prompt, return_tensors="pt").to(advanced_model.device)
            outputs = advanced_model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7)
            response_text = advanced_tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the response to get only the answer
            answer = response_text[len(prompt):].strip()
            return answer, doc_context

        else: # Normal Model (Local)
            response = llm_model.generate(prompt, max_tokens=512)
            return response, doc_context
            
    except Exception as e:
        return f"An error occurred: {str(e)}", None


# --- Streamlit App UI ---
st.set_page_config(page_title="Chat with your Data", page_icon="💬", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; }
    .stButton>button { background-color: #4285F4; color: white; border-radius: 20px; border: 1px solid #4285F4; }
    .stButton>button:hover { background-color: #FFFFFF; color: #4285F4; border: 1px solid #4285F4; }
    blockquote { background-color: #F1F3F4; border-left: 5px solid #4285F4; padding: 10px; margin: 10px 0px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'all_chats' not in st.session_state: st.session_state.all_chats = []


# --- Sidebar Controls ---
with st.sidebar:
    st.title("📄 Chat Controls")
    
    uploaded_file = st.file_uploader("Upload your data file", type=["docx", "pdf", "csv", "txt"])
    
    st.info("Note: The chatbot's knowledge is limited to the content of the uploaded document.", icon="💡")

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

    model_choice = st.selectbox(
        "Choose a model:",
        ('Normal Model (Local)', 'Advanced Model (DeepSeek)', 'Fast Model (Gemini)')
    )
    
    api_key = ""
    if model_choice == 'Fast Model (Gemini)':
        api_key = st.text_input("Enter Google AI API Key", type="password")
        st.markdown("For using the faster model you need to paste your Gemini API key here.")

    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("New Chat", use_container_width=True):
            if st.session_state.chat_history:
                st.session_state.all_chats.append(st.session_state.chat_history)
            st.session_state.chat_history = []
            st.rerun()

    with col2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
            
    if st.session_state.all_chats:
        st.markdown("---")
        st.subheader("Chat History")
        for i, chat in enumerate(st.session_state.all_chats):
            preview = chat[0]['content'] if chat and chat[0]['role'] == 'user' else "Empty Chat"
            if st.button(f"Chat {i+1}: {preview[:30]}...", key=f"history_{i}"):
                st.session_state.chat_history = chat


# --- Main Chat Interface ---
st.title("💬 Chatbot")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("context"):
            with st.expander("Show Sources"):
                st.markdown(f"> {message['context'].replace('---', '---')}")

if user_query := st.chat_input("Ask a question about your document..."):
    if not uploaded_file:
        st.warning("Please upload a document before asking questions.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking... (Advanced model may be slow on first run)"):
                response, context = ask_query(
                    user_query, model_choice, api_key, st.session_state.chat_history
                )
                st.markdown(response)
                if context:
                    with st.expander("Show Sources"):
                         st.markdown(f"> {context.replace('---', '---')}")
                
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": response, 
                    "context": context
                })

