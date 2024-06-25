import os
from dotenv import load_dotenv
from ragstack_colbert import CassandraDatabase, ColbertEmbeddingModel, ColbertVectorStore, ColbertRetriever
import json
import time

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
    indices = []
    for i, result in enumerate(results):
        indices.append(int(float(result[0].metadata['index'])))
    return indices

results = []
with open('../dataset/queries/holy_grail.txt', 'r') as file:
    start = time.time()
    for query in file.readlines():
        results.append(retrieve_nearest_documents(query))
    end = time.time()
print(f"Processing completed in {end - start:.2f} seconds")
storage_data = {"results": results}

with open('/Users/tiernan.lindauer/PycharmProjects/BenchmarkRAG/astra_results.json', 'w') as file:
    json.dump(storage_data, file)
