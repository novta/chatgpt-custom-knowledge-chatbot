from llama_index import GPTVectorStoreIndex, Settings, Document
from llama_index.readers.web import SimpleWebPageReader
from llama_index import StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
import os 
from apikey import apikey 

os.environ['OPENAI_API_KEY'] = apikey

def load_knowledge() -> list[Document]:
    # Load data from directory
    urls = ['https://www.areachica.se/']
    documents = SimpleWebPageReader(True).load_data(urls)
    return documents


def create_index() -> GPTVectorStoreIndex:
    print('Creating new index')
    # Load data
    documents = load_knowledge()
    # Create index from documents
    Settings.llm = OpenAI(model="gpt-4o-mini")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=1536)
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    Settings.num_output = 512
    Settings.context_window = 3900
    index = GPTVectorStoreIndex.from_documents(documents)
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


def query_index(index: GPTVectorStoreIndex):
    # Query index
    # query_engine = index.as_query_engine()
    while True:
        prompt = input("Type prompt...")
        query_engine = index.as_query_engine()
        response = query_engine.query(prompt)
        print(response)


def main():
    # Ask user if they want to refresh the index
    refresh_index = input("Do you want to refresh the index? (y/n) [n]: ")
    refresh_index = refresh_index.lower() == 'y'

    # If refreshing the index, create new index and save to file
    if refresh_index:
        index = create_index()
    # Otherwise, load index from file
    else:
        index = load_index()

    # Query index
    query_index(index)


if __name__ == '__main__':
    main()
