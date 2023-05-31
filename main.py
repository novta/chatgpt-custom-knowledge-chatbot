import os

from llama_index import GPTVectorStoreIndex
from bots.apikey import apikey
import streamlit as st


def initialize():
    # Store API key in environment variable
    os.environ["OPENAI_API_KEY"] = apikey
    # Prepare index for querying
    global index
    index = load_index()
    # Initialize global state for persistance between messages
    if "chat" not in st.session_state:
        st.session_state["chat"] = []


def render_app():
    st.title("🦾 Northprim chat bot 🤖")
    prompt_input = st.text_input("Ask me anything")

    if prompt_input:
        response = index.query(prompt_input)
        add_to_chat_history(prompt_input, response)


def render_chat_history():
    container_style = "\
        height: 400px; \
        overflow: auto; \
        display: flex; \
        flex-direction: column-reverse;\
    "
    chat_string = "<br>".join(st.session_state["chat"])
    st.write(
        f"<div style={container_style}><p>{chat_string}</p></div>",
        unsafe_allow_html=True,
    )


def add_to_chat_history(input, response):
    user_question = f"<span style='color: green;'>User: </span><span>{input}</span>"
    bot_reponse = f"<span style='color: blue;'>Bot: </span><span>{response}</span>"
    new_chat_value = [user_question, bot_reponse] + st.session_state["chat"]
    st.session_state["chat"] = new_chat_value


def load_index() -> GPTVectorStoreIndex:
    # Load index from file
    try:
        index = GPTVectorStoreIndex.load_from_disk("knowledge/index.json")
    except FileNotFoundError:
        st.write("<div style='color: red;'>No knowledge index found</div>")
    return index


def main():
    initialize()
    render_app()
    render_chat_history()


if __name__ == "__main__":
    main()
