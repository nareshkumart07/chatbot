import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# -----------------------------------------------------------
# 🔧 Load Flan-T5-Small Model (Locally)
# -----------------------------------------------------------
# As you requested, we are using AutoModelForSeq2SeqLM
model_id = "google/flan-t5-small"

@st.cache_resource(show_spinner="Loading chatbot model...")
def load_model():
    """
    Loads the Flan-T5-Small model and tokenizer locally.
    """
    # Use the specific classes you requested
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    return tokenizer, model

# Load the model and tokenizer
tokenizer, model = load_model()

# -----------------------------------------------------------
# 🧠 Generate Response Function (Local)
# -----------------------------------------------------------
def generate_response(prompt):
    """
    Generates a response from the local model given a user prompt.
    """
    try:
        # Tokenize the input prompt
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)

        # Generate the response
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,  # Increase token limit for chat
                num_beams=5,
                do_sample=True,      # Enable sampling for more varied chat responses
                temperature=0.9,
                top_p=0.95,
            )
        
        # Decode the generated tokens
        response_text = tokenizer.batch_decode(
            outputs, 
            skip_special_tokens=True
        )[0]
        
        return response_text

    except Exception as e:
        st.error(f"Error during generation: {e}")
        return "Sorry, an error occurred while generating a response."

# -----------------------------------------------------------
# 🎨 Streamlit UI (Chatbot App)
# -----------------------------------------------------------
st.set_page_config(page_title="Flan-T5 Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 Flan-T5 Chatbot (Local Model)")
st.write("Ask me anything! I'm a chatbot powered by a locally-loaded Flan-T5 model.")

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you today?"}
    ]

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input ---
if prompt := st.chat_input("What is on your mind?"):
    # 1. Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
            st.markdown(response)
    
    # 4. Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.caption("Powered by Google's Flan-T5-Small (Local).")
