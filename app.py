import streamlit as st
from chatbot_logic import get_file_content, setup_rag_pipeline, ask_query

# --- Streamlit App UI ---
st.set_page_config(page_title="Chat with your Data", page_icon="📄", layout="wide")

# Custom CSS for a cleaner look
st.markdown("""
<style>
    .stApp { background-color: #f9f9f9; }
    .stButton>button { 
        background-color: #4A90E2; 
        color: white; 
        border-radius: 20px; 
        border: 1px solid #4A90E2; 
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover { 
        background-color: #FFFFFF; 
        color: #4A90E2; 
        border: 1px solid #4A90E2; 
    }
    .st-emotion-cache-16txtl3 { padding-top: 2rem; }
    blockquote { 
        background-color: #f0f2f6; 
        border-left: 5px solid #4A90E2; 
        padding: 10px; 
        margin: 10px 0px; 
        border-radius: 5px; 
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'all_chats' not in st.session_state: st.session_state.all_chats = []

# --- Sidebar Controls ---
with st.sidebar:
    st.title("⚙️ Controls")
    st.markdown("---")

    st.header("Step 1: Add Your Document")
    uploaded_file = st.file_uploader(
        "Choose a file (.docx, .pdf, .csv, .txt)",
        type=["docx", "pdf", "csv", "txt"]
    )
    
    if uploaded_file:
        if "processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name:
            with st.spinner('Reading and preparing your file...'):
                file_content = get_file_content(uploaded_file)
                if file_content:
                    setup_rag_pipeline(file_content)
                    st.session_state.processed_file = uploaded_file.name
                    st.success("File ready!")
                else:
                    st.error("Could not read the file.")
    
    st.markdown("---")

    st.header("Step 2: Choose an AI Assistant")
    model_choice = st.selectbox(
        "Select the AI model you want to use:",
        ('Normal Model (Local)', 'Fast Model (Gemini)', 'BART')
    )
    
    # Add descriptions for each model to help non-technical users
    if model_choice == 'Normal Model (Local)':
        st.info("✅ Best for privacy. Runs on your computer but is a bit slower.", icon="🔒")
    elif model_choice == 'Fast Model (Gemini)':
        st.info("🚀 Fastest answers using Google's AI. Requires a key.", icon="⚡")
        api_key = st.text_input("Enter your Google AI API Key", type="password", help="You can get a free API key from Google AI Studio.")
    elif model_choice == 'BART':
        st.info("📝 Good for summarizing text and finding direct answers.", icon="✍️")

    st.markdown("---")
    
    st.header("Step 3: Manage Conversation")
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

# --- Main Chat Interface ---
st.title("📄 Chat With Your Document")

# Display a welcome message or the chat history
if not st.session_state.chat_history:
    st.markdown("### Welcome! I can answer questions about your document.")
    if not uploaded_file:
        st.info("To get started, please upload a file using the sidebar on the left.")
    else:
        st.success("Your document is loaded! Ask me anything about it in the chat box below.")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("context"):
            with st.expander("Show Sources"):
                st.markdown(f"> {message['context'].replace('---', '---')}")

# Chat input box at the bottom
if user_query := st.chat_input("Ask a question about your document..."):
    if not uploaded_file:
        st.warning("Please upload a document first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Use a placeholder for the API key if not needed for the selected model
                api_key_to_use = api_key if model_choice == 'Fast Model (Gemini)' else ""
                response, context = ask_query(
                    user_query, model_choice, api_key_to_use, st.session_state.chat_history
                )
                if response:
                    st.markdown(response)
                    if context:
                        with st.expander("Show Sources"):
                            st.markdown(f"> {context.replace('---', '---')}")
                    
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": response, 
                        "context": context
                    })

