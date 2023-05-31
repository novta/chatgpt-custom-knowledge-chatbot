# Bring in deps
import os 
from bots.apikey import apikey 

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, ServiceContext, Document

os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('🦜🔗 Smart bot')
prompt = st.text_input('Plug in your prompt here') 

def load_knowledge() -> list[Document]:
    # Load data from directory
    documents = SimpleDirectoryReader('knowledge').load_data()
    return documents

def create_index() -> GPTVectorStoreIndex:
    print('Creating new index')
    # Load data
    documents = load_knowledge()
    # Create index from documents
    service_context = ServiceContext.from_defaults(chunk_size_limit=3000)
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    save_index(index)
    return index


def save_index(index: GPTVectorStoreIndex):
    # Save index to file
    index.save_to_disk('knowledge/index.json')

def load_index() -> GPTVectorStoreIndex:
    # Load index from file
    try:
        index = GPTVectorStoreIndex.load_from_disk('knowledge/index.json')
    except FileNotFoundError:
        index = create_index()
    return index

# Show stuff to the screen if there's a prompt
index = load_index()

if prompt: 
    response = index.query(prompt)
    st.write(prompt) 
    st.write(response) 
