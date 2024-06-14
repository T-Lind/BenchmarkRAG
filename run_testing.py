import os
import time
import openai
import pinecone
import weaviate
from ragstack_colbert import CassandraDatabase, ColbertEmbeddingModel, ColbertVectorStore, ColbertRetriever
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_environment = os.getenv('PINECONE_ENVIRONMENT')
pinecone_index_name = os.getenv('PINECONE_INDEX_NAME')
astra_database_id = os.getenv('ASTRA_DATABASE_ID')
astra_token = os.getenv('ASTRA_TOKEN')


# Initialize Weaviate
weaviate_client = weaviate.Client("http://localhost:8080")

# AstraDB setup
keyspace = "default_keyspace"
database = CassandraDatabase.from_astra(astra_token=astra_token, database_id=astra_database_id, keyspace=keyspace)
embedding_model = ColbertEmbeddingModel()
vector_store = ColbertVectorStore(database=database, embedding_model=embedding_model)


# Function to load text files from a folder
def load_text_files(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r') as file:
                texts.append(file.read())
    return texts


# Function to get embeddings from OpenAI
def get_embeddings(texts):
    response = openai.Embedding.create(input=texts, model="")
    embeddings = [e['embedding'] for e in response['data']]
    return embeddings


# Function to perform hybrid search in Pinecone
def pinecone_hybrid_search(query_text, query_embedding):
    start_time = time.time()
    response = pinecone_index.query(
        query_embedding,
        top_k=10,
        include_metadata=True,
        query={"query": query_text}
    )
    latency = time.time() - start_time
    return response, latency


# Function to perform hybrid search in Weaviate
def weaviate_hybrid_search(query_text, query_embedding):
    start_time = time.time()
    response = weaviate_client.query \
        .get('YourClass', ['your', 'properties']) \
        .with_hybrid(query=query_text, alpha=0.5) \
        .with_near_vector({"vector": query_embedding}) \
        .do()
    latency = time.time() - start_time
    return response, latency


# Function to perform search in AstraDB
def astra_search(query_text):
    start_time = time.time()
    retriever = ColbertRetriever(vector_store=vector_store, embedding_model=embedding_model)
    response = retriever.retrieve(query_text, k=10)
    latency = time.time() - start_time
    return response, latency


# Function to calculate relevancy
def calculate_relevancy(query_embedding, search_results):
    result_embeddings = [res['embedding'] for res in search_results]
    relevancies = cosine_similarity([query_embedding], result_embeddings)
    avg_relevancy = relevancies.mean()
    return avg_relevancy


# Main function to compare the three vector stores
def compare_vector_stores(folder_path):
    texts = load_text_files(folder_path)
    embeddings = get_embeddings(texts)

    pinecone_latencies = []
    weaviate_latencies = []
    astra_latencies = []
    pinecone_relevancies = []
    weaviate_relevancies = []
    astra_relevancies = []

    for text, embedding in zip(texts, embeddings):
        pinecone_response, pinecone_latency = pinecone_hybrid_search(text, embedding)
        weaviate_response, weaviate_latency = weaviate_hybrid_search(text, embedding)
        astra_response, astra_latency = astra_search(text)

        pinecone_latencies.append(pinecone_latency)
        weaviate_latencies.append(weaviate_latency)
        astra_latencies.append(astra_latency)

        pinecone_relevancies.append(calculate_relevancy(embedding, pinecone_response['matches']))
        weaviate_relevancies.append(calculate_relevancy(embedding, weaviate_response['data']['Get']['YourClass']))
        astra_relevancies.append(calculate_relevancy(embedding, astra_response))

    avg_pinecone_latency = sum(pinecone_latencies) / len(pinecone_latencies)
    avg_weaviate_latency = sum(weaviate_latencies) / len(weaviate_latencies)
    avg_astra_latency = sum(astra_latencies) / len(astra_latencies)
    avg_pinecone_relevancy = sum(pinecone_relevancies) / len(pinecone_relevancies)
    avg_weaviate_relevancy = sum(weaviate_relevancies) / len(weaviate_relevancies)
    avg_astra_relevancy = sum(astra_relevancies) / len(astra_relevancies)

    print(f"Pinecone - Average Latency: {avg_pinecone_latency:.4f}s, Average Relevancy: {avg_pinecone_relevancy:.4f}")
    print(f"Weaviate - Average Latency: {avg_weaviate_latency:.4f}s, Average Relevancy: {avg_weaviate_relevancy:.4f}")
    print(f"AstraDB - Average Latency: {avg_astra_latency:.4f}s, Average Relevancy: {avg_astra_relevancy:.4f}")


# Run the comparison
compare_vector_stores('dataset/movie_scripts')
