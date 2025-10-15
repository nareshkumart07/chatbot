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
    """Loads the 'Normal' local GPT4All model."""
    # This model will be used for both 'Normal' and 'BART' modes,
    # with different prompts to guide its behavior.
    return GPT4All("orca-mini-3b-gguf2-q4_0.gguf")

@st.cache_resource
def load_encoder_model():
    """Loads the sentence transformer model for embeddings."""
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load models at startup
llm_model = load_llm_model()
encoder_model = load_encoder_model()


# --- File Reading and Processing ---

def read_docx(file):
    """Reads text from a .docx file."""
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def read_pdf(file):
    """Reads text from a .pdf file."""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def read_csv(file):
    """Reads content from a .csv file into a string."""
    return pd.read_csv(file).to_string()

def read_txt(file):
    """Reads text from a .txt file."""
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
    if not text:
        return []
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]

def setup_rag_pipeline(text_content):
    """Creates embeddings and a FAISS index for the document."""
    chunks = chunk_text(text_content)
    if not chunks:
        st.warning("Could not extract any text from the document. Please check the file content.")
        # Reset relevant session state if processing fails
        if 'faiss_index' in st.session_state: del st.session_state['faiss_index']
        if 'chunks' in st.session_state: del st.session_state['chunks']
        return
        
    embeddings = encoder_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    st.session_state.chunks = chunks
    st.session_state.faiss_index = index

def ask_query(query, model_type, api_key, chat_history):
    """
    Performs RAG to answer a query. It now supports three models and includes
    conversation history for better context.
    Returns the response and the document context used.
    """
    if 'faiss_index' not in st.session_state or 'chunks' not in st.session_state:
        return "Please upload and process a file first.", None

    query_emb = encoder_model.encode([query])
    _, I = st.session_state.faiss_index.search(np.array(query_emb), k=3)
    relevant_chunks = [st.session_state.chunks[i] for i in I[0]]
    doc_context = "\n\n---\n\n".join(relevant_chunks)

    # Prepare conversation history for the prompt
    history_str = ""
    recent_history = chat_history[-4:] # Use last 4 messages for context
    if recent_history:
        history_str += "Here is the recent conversation history:\n"
        for msg in recent_history:
            role = "User" if msg['role'] == 'user' else 'Assistant'
            history_str += f"{role}: {msg['content']}\n"
    
    # --- Prompt Engineering for Different Models ---
    
    # Generic Prompt for the Normal Local Model
    normal_prompt = f"""
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

    # Specialized Prompt to make the local model behave like BART (summarization-focused)
    bart_prompt = f"""
You are an expert AI assistant, similar to the BART model, specializing in summarizing and generating coherent, well-structured text. Your primary goal is to answer the user's question by summarizing the most relevant information from the provided document context and considering the conversation history.

{history_str}

Strictly use the following context from the document. If the answer is not in the provided context or history, you must state that you cannot find the answer in the document.

Document Context:
---
{doc_context}
---

User's Latest Question: {query}
Answer (provide a clear, concise summary based on the context):
"""

    try:
        if model_type == 'Fast Model (Gemini)':
            if not api_key:
                return "Error: Please enter your Google AI API key to use the Gemini model.", None
            genai.configure(api_key=api_key)
            gen_model = genai.GenerativeModel("gemini-pro")
            # Gemini uses the 'normal' prompt structure
            response = gen_model.generate_content(normal_prompt)
            return response.text, doc_context

        elif model_type == 'Advanced Model (BART)':
            # Use the local LLM with the BART-specific prompt
            response = llm_model.generate(bart_prompt, max_tokens=512)
            return response, doc_context
            
        else: # Normal Model (Local)
            # Use the local LLM with the standard prompt
            response = llm_model.generate(normal_prompt, max_tokens=512)
            return response, doc_context
            
    except Exception as e:
        error_message = f"An error occurred while communicating with the model: {str(e)}"
        st.error(error_message)
        return error_message, None


# --- Streamlit App UI ---
st.set_page_config(page_title="Chat with your Data", page_icon="💬", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #F0F2F6; }
    .stButton>button { background-color: #4A90E2; color: white; border-radius: 20px; border: 1px solid #4A90E2; transition: all 0.2s ease-in-out; }
    .stButton>button:hover { background-color: #FFFFFF; color: #4A90E2; border: 1px solid #4A90E2; }
    blockquote { background-color: #E8E8E8; border-left: 5px solid #4A90E2; padding: 10px; margin: 10px 0px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'all_chats' not in st.session_state: st.session_state.all_chats = []


# --- Sidebar Controls ---
with st.sidebar:
    st.title("📄 Chat Controls")
    
    uploaded_file = st.file_uploader("Upload your data file", type=["docx", "pdf", "csv", "txt"])
    
    st.info("Your chatbot's knowledge is limited to the content of your uploaded document.", icon="💡")

    if uploaded_file:
        # Process file only if it's new
        if "processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name:
            with st.spinner('Reading and indexing your file... This may take a moment.'):
                file_content = get_file_content(uploaded_file)
                if file_content:
                    setup_rag_pipeline(file_content)
                    st.session_state.processed_file = uploaded_file.name
                    # Clear chat history for the new document
                    st.session_state.chat_history = []
                    st.success("File processed! You can now ask questions.")
                else:
                    st.error("Failed to read the file. Please try another one.")

    # Model selection with the new BART option
    model_choice = st.selectbox(
        "Choose your model:",
        ('Normal Model (Local)', 'Fast Model (Gemini)', 'Advanced Model (BART)'),
        index=0 # Default to Normal Model
    )
    
    api_key = ""
    if model_choice == 'Fast Model (Gemini)':
        api_key = st.text_input("Enter Google AI API Key", type="password", help="Required for Gemini model.")
        st.markdown("[Get your Gemini API key here](https://aistudio.google.com/app/apikey)")

    st.markdown("---")
    
    # Chat management buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("New Chat", use_container_width=True):
            if st.session_state.chat_history: # Save the current chat before starting a new one
                st.session_state.all_chats.append(st.session_state.chat_history)
            st.session_state.chat_history = []
            st.rerun()

    with col2:
        if st.button("Clear History", use_container_width=True, type="secondary"):
            st.session_state.chat_history = []
            st.rerun()
            
    # Display previous chats for navigation
    if st.session_state.all_chats:
        st.markdown("---")
        st.subheader("Previous Chats")
        for i, chat in enumerate(st.session_state.all_chats):
            # Use the first user message as a preview
            preview = chat[0]['content'] if chat and chat[0]['role'] == 'user' else "Chat"
            if st.button(f"Chat {i+1}: {preview[:30]}...", key=f"history_{i}", use_container_width=True):
                st.session_state.chat_history = chat
                st.rerun()

# --- Main Chat Interface ---
st.title("💬 Chat With Your Document")
st.markdown("Upload a document and choose a model from the sidebar to begin.")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("context"):
            with st.expander("Show document context used for this answer"):
                st.markdown(f"> {message['context'].replace('---', '---')}")

if user_query := st.chat_input("Ask a question about your document..."):
    if "processed_file" not in st.session_state or not st.session_state.processed_file:
        st.warning("Please upload and process a document before asking questions.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, context = ask_query(
                    user_query, model_choice, api_key, st.session_state.chat_history
                )
                if response:
                    # Streamlit's markdown renderer will format the text
                    st.markdown(response)
                    if context:
                        with st.expander("Show document context"):
                             st.markdown(f"> {context.replace('---', '---')}")
                    
                    # Add the complete response to history
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": response, 
                        "context": context
                    })
