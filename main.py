import os
from apikey import apikey
import streamlit as st
from app import load_index

# Store API key in environment variable
os.environ["OPENAI_API_KEY"] = apikey

if not hasattr(st.session_state, 'chat'):
    st.session_state['chat'] = []

index = load_index()

st.title('🦾 Northprim chat bot 🤖')
prompt_input = st.text_input("Ask me anything")

def render_chat_history():
    container_style = "\
        height: 400px; \
        overflow: auto; \
        display: flex; \
        flex-direction: column-reverse;\
    "
    chat_string = "<br>".join(st.session_state['chat'])
    st.write(f"<div style={container_style}><p>{chat_string}</p></div>", unsafe_allow_html=True)

def add_to_chat_history(input, response): 
    user_question = f"<span style='color: green;'>User: </span><span>{input}</span>"
    bot_reponse = f"<span style='color: blue;'>Bot: </span><span>{response}</span>"
    new_chat_value = [user_question, bot_reponse] + st.session_state['chat']
    st.session_state['chat'] = new_chat_value

if prompt_input:
    response = index.query(prompt_input)
    add_to_chat_history(prompt_input, response)
    print(st.session_state['chat'])

render_chat_history()