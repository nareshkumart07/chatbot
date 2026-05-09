<div align="center">

<img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
<img src="https://img.shields.io/badge/FAISS-Vector%20Store-009688?style=for-the-badge" alt="FAISS">
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">

<br/><br/>

<h1>💬 Chat with Your Data</h1>

<p><strong>An interactive Streamlit application to converse with your documents using multiple powerful LLMs and a RAG pipeline.</strong></p>

<p>
  <a href="#-features">Features</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-getting-started">Getting Started</a> •
  <a href="#-usage">Usage</a>
</p>

</div>

---

## ✨ Features

<table>
  <tr>
    <td>📁 <strong>Multiple File Formats</strong></td>
    <td>Upload and process <code>.docx</code>, <code>.pdf</code>, <code>.csv</code>, and <code>.txt</code> files seamlessly.</td>
  </tr>
  <tr>
    <td>🤖 <strong>Triple LLM Support</strong></td>
    <td>Choose from three different language models depending on your speed, privacy, and accuracy needs.</td>
  </tr>
  <tr>
    <td>🔍 <strong>RAG Pipeline</strong></td>
    <td>Retrieval-Augmented Generation using FAISS for contextually-aware, document-grounded answers.</td>
  </tr>
  <tr>
    <td>💬 <strong>Interactive Chat UI</strong></td>
    <td>Clean chat interface to ask questions and receive instant answers from your documents.</td>
  </tr>
  <tr>
    <td>🗂️ <strong>Chat Management</strong></td>
    <td>New chat, clear chat, and persistent chat history — all from the sidebar.</td>
  </tr>
</table>

---

## 🤖 Available Models

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Type</th>
      <th>Speed</th>
      <th>Privacy</th>
      <th>Requirements</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>🏠 <strong>Normal Model (Local)</strong></td>
      <td>GPT4All (<code>orca-mini-3b</code>)</td>
      <td>🐢 Slower</td>
      <td>🔒 Full Privacy</td>
      <td>None — runs on-device</td>
    </tr>
    <tr>
      <td>⚡ <strong>Fast Model (Gemini)</strong></td>
      <td>Google Gemini API</td>
      <td>🚀 Fast</td>
      <td>☁️ Cloud</td>
      <td>Google AI API Key</td>
    </tr>
    <tr>
      <td>🧠 <strong>BART</strong></td>
      <td>Hugging Face Transformer</td>
      <td>⚡ Medium</td>
      <td>🔒 Local</td>
      <td>None — runs on-device</td>
    </tr>
  </tbody>
</table>

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Framework** | [Streamlit](https://streamlit.io/) |
| **LLMs** | GPT4All (local) · Google Gemini (API) · BART (Hugging Face) |
| **Embeddings** | [Sentence-Transformers](https://www.sbert.net/) |
| **Vector Store** | [FAISS](https://faiss.ai/) — Facebook AI Similarity Search |
| **File Handling** | `python-docx` · `PyPDF2` · `pandas` |

---

## 🚀 Getting Started

### Prerequisites

- Python **3.8** or higher
- `pip` package manager

---

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-project-folder>
```

---

### 2. Create a Virtual Environment *(Recommended)*

<table>
<tr>
<th>Windows</th>
<th>macOS / Linux</th>
</tr>
<tr>
<td>

```bash
python -m venv venv
venv\Scripts\activate
```

</td>
<td>

```bash
python3 -m venv venv
source venv/bin/activate
```

</td>
</tr>
</table>

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first time you run the app, the local GPT4All model (`orca-mini-3b-gguf2-q4_0.gguf`) will be downloaded automatically. This is a **one-time process**.

---

### 4. Run the Application

```bash
streamlit run app.py
```

Your browser will automatically open a new tab with the app running at `http://localhost:8501`.

---

## 📖 Usage

Follow these steps after launching the app:

<table>
  <tr>
    <th>Step</th>
    <th>Action</th>
    <th>Details</th>
  </tr>
  <tr>
    <td align="center"><strong>1</strong></td>
    <td>📤 <strong>Upload a Document</strong></td>
    <td>Use the file uploader in the sidebar to select a <code>.docx</code>, <code>.pdf</code>, <code>.csv</code>, or <code>.txt</code> file.</td>
  </tr>
  <tr>
    <td align="center"><strong>2</strong></td>
    <td>⏳ <strong>Wait for Processing</strong></td>
    <td>The app reads the file, chunks the text, creates embeddings, and builds a FAISS vector index. A success message confirms it's ready.</td>
  </tr>
  <tr>
    <td align="center"><strong>3</strong></td>
    <td>🤖 <strong>Choose a Model</strong></td>
    <td>Select <em>Normal (Local)</em>, <em>Fast (Gemini)</em>, or <em>BART</em> from the dropdown in the sidebar.</td>
  </tr>
  <tr>
    <td align="center"><strong>4</strong></td>
    <td>🔑 <strong>Add API Key</strong> <em>(if needed)</em></td>
    <td>If you select the <strong>Fast Model</strong>, paste your Google AI API key into the input field that appears.</td>
  </tr>
  <tr>
    <td align="center"><strong>5</strong></td>
    <td>💬 <strong>Start Chatting</strong></td>
    <td>Type your questions into the chat input at the bottom of the screen and press <kbd>Enter</kbd>.</td>
  </tr>
</table>

---

## 📂 Project Structure

```
📦 your-project-folder
├── app.py               # Main Streamlit application
├── chat_bot.py          # LLM logic and RAG pipeline
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m 'Add some AmazingFeature'`
4. Push to the branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <p>Made with ❤️ using Streamlit &amp; Python</p>
</div>
