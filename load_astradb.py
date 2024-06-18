# Some minor setup for the notebook
import warnings
import nest_asyncio

# Ignore all warnings
warnings.filterwarnings("ignore")

# Allows for running async code in Jupyter notebooks
nest_asyncio.apply()

import os
import csv
import time
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ragstack_colbert import CassandraDatabase, ColbertEmbeddingModel, ColbertVectorStore
from datasets import load_dataset

load_dotenv()

bioasq = load_dataset("rag-datasets/rag-mini-bioasq", "question-answer-passages")['test']
corpus = load_dataset("rag-datasets/rag-mini-bioasq", "text-corpus")['passages']

astra_database_id = os.getenv('ASTRA_DATABASE_ID')
astra_token = os.getenv('ASTRA_TOKEN')
keyspace = "benchmarking3"

database = CassandraDatabase.from_astra(astra_token=astra_token, database_id=astra_database_id, keyspace=keyspace)
embedding_model = ColbertEmbeddingModel()
vector_store = ColbertVectorStore(database=database, embedding_model=embedding_model)


def store_in_astra(chunks):
    metadatas = []
    for doc in chunks:
        metadatas.append({"date": time.strftime("%m/%d/%Y"), 'index': doc['id']})
    vector_store.add_texts(texts=[doc['passage'] for doc in chunks], metadatas=metadatas, doc_id="rag-bio-test")


store_in_astra(corpus)
