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
import mysql.connector

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


# --- Database Connection ---

# Establish connection to MySQL.
# Uses st.cache_resource to only run once.
@st.cache_resource
def init_connection():
    try:
        return mysql.connector.connect(**st.secrets["mysql"])
    except mysql.connector.Error as err:
        st.error(f"Error connecting to MySQL: {err}")
        return None

# Perform query.
# Uses st.cache_data to only rerun when the query changes or after 10 min.
@st.cache_data(ttl=600)
def run_query(query):
    conn = init_connection()
    if conn:
        with conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchall()
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
        st.warning("Could not extract text from the database.")
        return
        
    embeddings = encoder_model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    
    st.session_state.chunks = chunks
    st.session_state.faiss_index = index
    st.session_state.processed_db = True
    st.success("Database content processed successfully!")


def ask_query(query, model_type, api_key):
    """
    Performs RAG to answer a query.
    """
    if 'faiss_index' not in st.session_state or 'chunks' not in st.session_state:
        return "Please connect to the database and process the data first."

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
            gen_model = genai.GenerativeModel("gemini-2.0-flash")
            response = gen_model.generate_content(prompt)
            return response.text
        else: # Normal Model (Local)
            response = llm_model.generate(prompt, max_tokens=512)
            return response
    except Exception as e:
        return f"An error occurred: {str(e)}"

# --- Streamlit App UI ---

st.set_page_config(page_title="Chat with your Data", page_icon="💬", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #F0F8FF; }
    .stButton>button { background-color: #4F8BF9; color: white; border-radius: 20px; border: 1px solid #4F8BF9; }
    .stButton>button:hover { background-color: #FFFFFF; color: #4F8BF9; border: 1px solid #4F8BF9; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'all_chats' not in st.session_state: st.session_state.all_chats = []
if 'processed_db' not in st.session_state: st.session_state.processed_db = False


# --- Sidebar Controls ---
with st.sidebar:
    st.title("📄 Chat with your Data")
    st.markdown("Connect to your MySQL database to ask questions.")

    table_name = st.text_input("Enter the table name to chat with:")

    if st.button("Connect and Process Data"):
        if table_name:
            with st.spinner("Connecting to database and fetching data..."):
                try:
                    # Fetch data from the specified table
                    rows = run_query(f"SELECT * FROM {table_name};")
                    if rows:
                        # Convert fetched data to a string format for the RAG pipeline
                        db_content = pd.DataFrame(rows).to_string()
                        setup_rag_pipeline(db_content)
                    else:
                        st.warning("No data found in the specified table.")
                except Exception as e:
                    st.error(f"Failed to process database table: {e}")
        else:
            st.warning("Please enter a table name.")


    model_choice = st.selectbox("Choose a model:", ('Normal Model (Local)', 'Fast Model (Gemini)'))
    
    api_key = ""
    if model_choice == 'Fast Model (Gemini)':
        st.markdown("For using faster model you need to paste gemini api key here.")
        api_key = st.text_input("Enter your Google AI API Key", type="password")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("New Chat", use_container_width=True):
            if st.session_state.chat_history:
                first_q = st.session_state.chat_history[0]['content']
                chat_title = (first_q[:30] + '...') if len(first_q) > 30 else first_q
                st.session_state.all_chats.append({"title": chat_title, "messages": st.session_state.chat_history})
            st.session_state.chat_history = []
            st.rerun()

    with col2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    st.markdown("### Chat History")
    if not st.session_state.all_chats:
        st.write("No past chats saved.")
    else:
        for i, saved_chat in enumerate(st.session_state.all_chats):
            if st.button(saved_chat['title'], key=f"history_chat_{i}"):
                st.session_state.chat_history = saved_chat['messages']
                st.rerun()

    st.markdown("""
    ---
    **Model Information:**
    - **Normal Model:** Runs locally. Slower, but private.
    - **Fast Model:** Uses Gemini API. Faster, requires API key.
    """)


# --- Main Chat Interface ---
st.title("💬 Chatbot")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if user_query := st.chat_input("Ask a question about your database..."):
    if not st.session_state.processed_db:
        st.warning("Please connect to the database before asking questions.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ask_query(user_query, model_choice, api_key)
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

