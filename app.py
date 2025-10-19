import streamlit as st
from chat_bot import get_file_content, setup_rag_pipeline, ask_query

# --- Streamlit App UI Configuration ---
st.set_page_config(page_title="Chat with your Data", page_icon="💬", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; }
    .stButton>button { background-color: #4285F4; color: white; border-radius: 20px; border: 1px solid #4285F4; }
    .stButton>button:hover { background-color: #FFFFFF; color: #4285F4; border: 1px solid #4285F4; }
    blockquote { background-color: #F1F3F4; border-left: 5px solid #4285F4; padding: 10px; margin: 10px 0px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'all_chats' not in st.session_state:
    st.session_state.all_chats = []


# --- Sidebar Controls ---
with st.sidebar:
    st.title("📄 Chat Controls")
    
    uploaded_file = st.file_uploader("Upload your data file", type=["docx", "pdf", "csv", "txt"])
    
    st.info("Note: The chatbot's knowledge is limited to the content of the uploaded document.", icon="💡")

    # Process file on upload
    if uploaded_file:
        if "processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name:
            with st.spinner('Reading and indexing your file... This may take a moment.'):
                file_content = get_file_content(uploaded_file)
                if file_content:
                    setup_rag_pipeline(file_content)
                    st.session_state.processed_file = uploaded_file.name
                    st.session_state.chat_history = [] # Clear history for new file
                    st.session_state.all_chats = [] # Clear all history for new file
                    st.success("File processed successfully!")
                else:
                    st.error("Failed to read or process the file.")

    model_choice = st.selectbox(
        "Choose a model:",
        ('Normal Model (Local)', 'Fast Model (Gemini)')
    )
    
    api_key = ""
    if model_choice == 'Fast Model (Gemini)':
        api_key = st.text_input("Enter Google AI API Key", type="password", help="You can get your key from Google AI Studio.")
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
            
    # Display past chat sessions for selection
    if st.session_state.all_chats:
        st.markdown("---")
        st.subheader("Chat History")
        for i, chat in enumerate(st.session_state.all_chats):
            preview = chat[0]['content'] if chat and chat[0]['role'] == 'user' else "Empty Chat"
            if st.button(f"Chat {i+1}: {preview[:30]}...", key=f"history_{i}", use_container_width=True):
                st.session_state.chat_history = chat
                st.rerun()


# --- Main Chat Interface ---
st.title("💬 Chatbot")
st.write("Upload a document and start asking questions!")

# Display current chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show sources only for assistant messages that have them
        if message["role"] == "assistant" and message.get("context"):
            with st.expander("Show Sources"):
                # Format context for better readability
                st.markdown(f"> {message['context'].replace('---', '---')}")

# Handle new user input
if user_query := st.chat_input("Ask a question about your document..."):
    if not uploaded_file:
        st.warning("Please upload a document before asking questions.")
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
                    st.markdown(response)
                    if context:
                        with st.expander("Show Sources"):
                             st.markdown(f"> {context.replace('---', '---')}")
                    
                    # Append the full response to history
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": response, 
                        "context": context
                    })
