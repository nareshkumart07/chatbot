<div align="center">

<img src="https://img.shields.io/badge/Python-3-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
<img src="https://img.shields.io/badge/FAISS-Vector%20Store-009688?style=for-the-badge" alt="FAISS">

<br/><br/>

<h1>💬 Chat with Your Data</h1>

<p><strong>A Streamlit document Q&amp;A demonstration with FAISS retrieval and a choice of local or API-based text generation.</strong></p>

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
    <td>Choose GPT4All, the Gemini API, or BART. See the processing locations and setup requirements below.</td>
  </tr>
  <tr>
    <td>🔍 <strong>RAG Pipeline</strong></td>
    <td>Retrieval-Augmented Generation using FAISS for contextually-aware, document-grounded answers.</td>
  </tr>
  <tr>
    <td>💬 <strong>Interactive Chat UI</strong></td>
    <td>Ask questions and inspect the retrieved document context alongside generated answers.</td>
  </tr>
  <tr>
    <td>🗂️ <strong>Chat Management</strong></td>
    <td>New chat and clear chat controls, with conversation state held in the current Streamlit session. History is not saved to a database or file.</td>
  </tr>
</table>

---

## 🤖 Available Models

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Type</th>
      <th>Processing location</th>
      <th>Requirements</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>🏠 <strong>Normal Model (Local)</strong></td>
      <td>GPT4All (<code>orca-mini-3b</code>)</td>
      <td>Machine hosting Streamlit</td>
      <td>Download of <code>orca-mini-3b-gguf2-q4_0.gguf</code>; sufficient local memory and disk</td>
    </tr>
    <tr>
      <td>⚡ <strong>Fast Model (Gemini)</strong></td>
      <td>Google Gemini API</td>
      <td>Google's Gemini API</td>
      <td>Google AI API key, network access, and access to <code>gemini-2.5-flash</code></td>
    </tr>
    <tr>
      <td>🧠 <strong>BART</strong></td>
      <td>Hugging Face Transformer</td>
      <td>Machine hosting Streamlit</td>
      <td>Download of <code>facebook/bart-large-cnn</code>; sufficient local memory and disk</td>
    </tr>
  </tbody>
</table>

The UI labels “Normal” and “Fast” identify model options; response speed and answer quality have not been benchmarked in this repository. Embeddings use `all-MiniLM-L6-v2` on the machine hosting Streamlit. Selecting Gemini sends the question, retrieved document text, and recent conversation history to Google's API. Local generation does not make a remotely hosted Streamlit deployment equivalent to processing on your own computer.

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

- Python 3.11 and the `pip` package manager. Python 3.11 matches the image configured in `.devcontainer/devcontainer.json`.
- Internet access for installing dependencies and downloading model files on the first launch.
- Sufficient memory and disk space for GPT4All, the embedding model, and BART. All three are loaded at startup, even when you intend to use only Gemini.

Dependencies are currently unpinned, and the repository does not provide a lockfile or compatibility test results. The commands below match the repository layout; a clean-environment installation and runtime validation are still needed.

---

### 1. Clone the Repository

```bash
git clone https://github.com/nareshkumart07/chatbot.git
cd chatbot
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
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> **First launch:** `chatbot_logic.py` initializes GPT4All (`orca-mini-3b-gguf2-q4_0.gguf`), Sentence-Transformers (`all-MiniLM-L6-v2`), and BART (`facebook/bart-large-cnn`) during import. Missing model files may be downloaded and cached by their libraries. Startup therefore requires network access until the models are cached and can take time before the UI becomes available. Model download or initialization failures can prevent the app from starting.

---

### 4. Run the Application

```bash
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

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
📦 chatbot
├── app.py               # Main Streamlit application
├── chatbot_logic.py     # File readers, model loading, retrieval, and generation
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## Implementation and Current Limits

- Text is split into 500-character chunks with a 50-character overlap. `all-MiniLM-L6-v2` creates embeddings, and a FAISS `IndexFlatL2` retrieves three chunks for each question.
- The document chunks, vector index, and chat history live in Streamlit session state. This implementation does not persist them across new sessions.
- PDF extraction reads existing text with `PyPDF2`; scanned PDFs need OCR before upload. TXT files must be UTF-8 encoded, and CSV files are converted to text for retrieval.
- “Show Sources” displays retrieved text, rather than page-level citations. Generated answers still need checking against the document.
- The repository does not currently include an automated test suite, benchmark results, or a deployment guide. It is a learning and portfolio demonstration, with runtime and model-quality validation still needed.

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

No `LICENSE` file is currently included in this repository.

---

<div align="center">
  <p>Made with ❤️ using Streamlit &amp; Python</p>
</div>
