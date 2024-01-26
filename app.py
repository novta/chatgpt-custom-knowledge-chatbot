# Bring in deps
import os 
from apikey import apikey 

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, ServiceContext, Document
from llama_index.readers.web import SimpleWebPageReader
from llama_index import StorageContext, load_index_from_storage

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
    # rebuild storage context
    storage_context = StorageContext.from_defaults(chunk_size_limit=3000, persist_dir='knowledge')
    # load index
    index = load_index_from_storage(storage_context)
    save_index(index)
    return index


def save_index(index: GPTVectorStoreIndex):
    # Save index to file
    index.storage_context.persist('knowledge')

def load_index() -> GPTVectorStoreIndex:
    # Load index from file
    try:
        # rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir='knowledge')
        # load index
        index = load_index_from_storage(storage_context)
    except FileNotFoundError:
        index = create_index()
    return index

# Show stuff to the screen if there's a prompt
index = load_index()

if prompt:
    query_engine = index.as_query_engine()
    response = query_engine.query(prompt)
    st.write(prompt) 
    st.write(response) 
