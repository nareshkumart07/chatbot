Chat with Your Data - Streamlit Application

This interactive web application, built with Streamlit, allows you to chat with your documents. You can upload a file (supporting .docx, .pdf, .csv, and .txt formats) and ask questions about its content using one of three powerful language models.

✨ Features

Multiple File Formats: Upload and process text from .docx, .pdf, .csv, and .txt files.

Triple LLM Models:

Normal Model (Local): A private and secure GPT4All model that runs directly on your machine. It's slower but requires no internet connection or API keys.

Fast Model (Gemini): Utilizes Google's powerful Gemini API for quicker responses. Requires a Google AI API key.

BART: A transformer-based model from Hugging Face, great for summarization and question-answering tasks.

RAG Pipeline: Implements a Retrieval-Augmented Generation (RAG) pipeline using FAISS for efficient and contextually-aware answers based on your document's content.

Interactive Chat Interface: A user-friendly chat interface to ask questions and receive answers.

Chat Management:

New Chat: Start a fresh conversation.

Clear Chat: Clear the current conversation history.

Chat History: Save and revisit previous conversations.

🛠️ Tech Stack

Framework: Streamlit

LLMs: GPT4All (local), Google Gemini (API), BART (Hugging Face)

Embeddings: Sentence-Transformers

Vector Store: FAISS (Facebook AI Similarity Search)

File Handling: python-docx, PyPDF2, pandas

🚀 Getting Started
Follow these steps to set up and run the application on your local machine.

Prerequisites

Python 3.8 or higher

pip package manager

1. Clone the Repository

First, get the project files onto your machine. If you're using git, you can clone the repository. Otherwise, download the files (app.py, chat_bot.py, and requirements.txt) into a new folder.

git clone <your-repository-url>
cd <your-project-folder>


2. Create a Virtual Environment (Recommended)

It's a good practice to create a virtual environment to manage project dependencies.

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate


3. Install Dependencies

Install all the required Python libraries using the requirements.txt file.

pip install -r requirements.txt


The first time you run the app, the local GPT4All model (orca-mini-3b-gguf2-q4_0.gguf) will be downloaded. This is a one-time process.

4. Run the Application

Launch the Streamlit app with the following command:

streamlit run app.py


Your web browser will automatically open a new tab with the application running.

Usage

Upload a Document: Use the file uploader in the sidebar to select a .docx, .pdf, .csv, or .txt file.

Wait for Processing: The app will read the file, chunk the text, create embeddings, and build a vector index. A success message will appear when it's ready.

Choose a Model: Select either the "Normal Model (Local)", "Fast Model (Gemini)", or "BART" from the dropdown.

(Optional) Add API Key: If you choose the "Fast Model," an input field will appear. Paste your Google AI API key here.

Start Chatting: Type your questions into the chat input at the bottom of the main screen and press Enter.
