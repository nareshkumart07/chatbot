import streamlit as st
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
from transformers import BartTokenizer, BartForConditionalGeneration

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
def load_bart_model():
    """Loads the BART model and tokenizer from Hugging Face."""
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    return tokenizer, model

# Load models at startup
llm_model = load_llm_model()
encoder_model = load_encoder_model()
bart_tokenizer, bart_model = load_bart_model()


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
    Performs RAG to answer a query using the selected model and conversation history.
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
            gen_model = genai.GenerativeModel("gemini-2.5-flash")
            response = gen_model.generate_content(prompt)
            return response.text, doc_context

        elif model_type == 'BART':
            inputs = bart_tokenizer(prompt, return_tensors='pt', max_length=1024, truncation=True)
            summary_ids = bart_model.generate(inputs['input_ids'], num_beams=4, max_length=256, early_stopping=True)
            response = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return response, doc_context

        else: # Normal Model (Local)
            response = llm_model.generate(prompt, max_tokens=512)
            return response, doc_context
            
    except Exception as e:
        error_message = f"An error occurred while communicating with the model API: {str(e)}"
        st.error(error_message)
        return error_message, None

