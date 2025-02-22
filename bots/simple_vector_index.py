from llama_index.core import GPTVectorStoreIndex, SimpleDirectoryReader, Settings, Document
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
import logging
import os
from apikey import apikey
os.environ['OPENAI_API_KEY'] = apikey

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_knowledge() -> list[Document]:
    documents = SimpleDirectoryReader('knowledge').load_data()
    return documents

def create_index() -> GPTVectorStoreIndex:
    print('Creating new index')
    documents = load_knowledge()
    try:
        # Initialize embedding model with proper parameters
        embed_model = OpenAIEmbedding(
            model="text-embedding-ada-002",
            embed_batch_size=100,
            api_key=os.environ.get('OPENAI_API_KEY'),
            max_embeddings_per_request=20
        )
        
        # Verify initialization
        logger.info("Embedding model initialized successfully")
        
        # Configure settings
        Settings.llm = OpenAI(model="gpt-4o-mini")
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
        Settings.num_output = 512
        Settings.context_window = 3900
        
        # Create index with the embedding model directly
        logger.info(f"Creating index from {len(documents)} documents...")
        index = GPTVectorStoreIndex.from_documents(
            documents,
            persist_dir='knowledge',
            embedding_model=embed_model,
            embedding_batch_size=embed_model.embed_batch_size
        )
        save_index(index)
        return index
        
    except Exception as e:
        logger.error(f"Failed to create index: {str(e)}")
        raise

def save_index(index: GPTVectorStoreIndex):
    index.storage_context.persist('knowledge')

def load_index() -> GPTVectorStoreIndex:
    try:
        storage_context = StorageContext.from_defaults(persist_dir='knowledge')
        index = load_index_from_storage(storage_context)
    except FileNotFoundError:
        index = create_index()
    return index

def query_index(index: GPTVectorStoreIndex):
    while True:
        prompt = input("Type prompt...")
        query_engine = index.as_query_engine()
        response = query_engine.query(prompt)
        print(response)

def main():
    refresh_index = input("Do you want to refresh the index? (y/n) [n]: ")
    refresh_index = refresh_index.lower() == 'y'
    if refresh_index:
        index = create_index()
    else:
        index = load_index()
    query_index(index)

if __name__ == '__main__':
    main()