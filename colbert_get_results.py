import os
from dotenv import load_dotenv
from ragstack_colbert import CassandraDatabase, ColbertEmbeddingModel, ColbertVectorStore, ColbertRetriever

load_dotenv()

astra_database_id = os.getenv('ASTRA_DATABASE_ID')
astra_token = os.getenv('ASTRA_TOKEN')
keyspace = "benchmarking"

database = CassandraDatabase.from_astra(astra_token=astra_token, database_id=astra_database_id, keyspace=keyspace)
embedding_model = ColbertEmbeddingModel()
vector_store = ColbertVectorStore(database=database, embedding_model=embedding_model)

retriever = ColbertRetriever(embedding_model=embedding_model, database=database)


def retrieve_nearest_documents(query, k=5):
    results = retriever.text_search(query, k=k)
    for i, result in enumerate(results):
        print(int(float(result[0].metadata['index'])))


if __name__ == "__main__":
    query = input("Enter your query: ")
    retrieve_nearest_documents(query)
