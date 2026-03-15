from google import genai
import os
import json
import time


import streamlit as st
from google import genai

# --- Configuration & Initialization ---
st.set_page_config(page_title="Gemini Chat", page_icon="🤖")
st.title("Gemini 3.0 Free Chat")

# Input for API Key (or set as environment variable for security)
api_key = st.sidebar.text_input("", type="password")

if api_key:
    client = genai.Client(api_key=api_key)
    
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history from session state
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is on your mind?"):
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate Gemini response
        try:
            # We use Gemini 3 Flash as it's the fastest free-tier model in 2026
            response = client.models.generate_content(
                model="gemini-3-flash-preview", 
                contents=prompt
            )
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response.text)
            
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Please enter your API Key in the sidebar to start chatting.")