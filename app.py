import streamlit as st
import time
from docx import Document
import pandas as pd
import PyPDF2
from io import StringIO
import os

# Dummy functions to simulate the backend logic from your script
# In a real application, you would integrate your actual model and functions here.

def read_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
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

def ask_query(query, model_type, file_content):
    """
    Simulates asking a query to the selected model.
    In a real app, this would involve text chunking, embedding, and the RAG architecture.
    """
    if not file_content:
        return "Please upload a data file first."

    # Simulate a delay to mimic model processing time
    if model_type == 'Fast Model':
        time.sleep(1)
        return f"Fast model response for '{query}' based on the document."
    else: # Normal Model
        time.sleep(3)
        return f"Normal model response for '{query}' based on the document."


# --- Streamlit App ---

st.set_page_config(page_title="Chat with your Data", page_icon="💬", layout="wide")

# Custom CSS for the blue and white theme
st.markdown("""
<style>
    .stApp {
        background-color: #F0F8FF;
    }
    .st-emotion-cache-16txtl3 {
        padding: 2rem 1rem 1rem;
    }
    .st-emotion-cache-1y4p8pa {
        width: 100%;
        padding: 1rem 1rem 1rem;
        max-width: 80rem;
    }
    .st-emotion-cache-1avcm0n {
        background-color: #FFFFFF;
    }
    .st-emotion-cache-1c7y2kd {
        border-bottom: 1px solid #4F8BF9;
    }
    .stButton>button {
        background-color: #4F8BF9;
        color: white;
        border-radius: 20px;
        border: 1px solid #4F8BF9;
        padding: 10px 24px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #FFFFFF;
        color: #4F8BF9;
        border: 1px solid #4F8BF9;
    }
    .st-emotion-cache-1v0mbdj {
        border-radius: 10px;
        border: 1px solid #4F8BF9;
    }
    .st-emotion-cache-q8f62p {
        background-color: #E6F0FF;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_fast_model_request_time' not in st.session_state:
    st.session_state.last_fast_model_request_time = 0
if 'fast_model_request_count' not in st.session_state:
    st.session_state.fast_model_request_count = 0
if 'file_content' not in st.session_state:
    st.session_state.file_content = None


# --- Sidebar for controls ---
with st.sidebar:
    st.title("📄 Chat with your Data")
    st.markdown("Upload a document and ask questions about its content.")

    uploaded_file = st.file_uploader("Upload your data file", type=["docx", "pdf", "csv", "txt"])
    if uploaded_file:
        if st.session_state.file_content is None: # Process file only once
             with st.spinner('Processing file...'):
                st.session_state.file_content = get_file_content(uploaded_file)
                if st.session_state.file_content:
                    st.success("File processed successfully!")

    model_choice = st.selectbox("Choose a model:", ('Fast Model', 'Normal Model'))
    st.markdown("""
    **Model Information:**
    - **Fast Model:** Quicker responses, but limited to 2 questions per minute.
    - **Normal Model:** Slower responses, with no usage limits.
    """)


# --- Main Chat Interface ---
st.title("💬 Chatbot")

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
user_query = st.chat_input("Ask a question about your document...")

if user_query:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_query)

    # Check for fast model rate limit
    is_rate_limited = False
    if model_choice == 'Fast Model':
        current_time = time.time()
        if current_time - st.session_state.last_fast_model_request_time < 60:
            if st.session_state.fast_model_request_count >= 2:
                is_rate_limited = True
                wait_time = 60 - (current_time - st.session_state.last_fast_model_request_time)
                st.warning(f"Fast model limit reached. Please wait {int(wait_time)} seconds.")
            else:
                st.session_state.fast_model_request_count += 1
        else:
            st.session_state.last_fast_model_request_time = current_time
            st.session_state.fast_model_request_count = 1

    if not is_rate_limited:
        with st.spinner("Thinking..."):
            # Get response from the backend
            response = ask_query(user_query, model_choice, st.session_state.file_content)

            # Add assistant message to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(response)

# To run this app:
# 1. Save the code as chatbot_dashboard.py
# 2. Make sure you have streamlit installed (`pip install streamlit`)
# 3. In your terminal, run: `streamlit run chatbot_dashboard.py`
