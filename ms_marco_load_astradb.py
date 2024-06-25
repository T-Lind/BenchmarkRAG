from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

ms_marco = load_dataset("microsoft/ms_marco", "v2.1")

subset = ms_marco['train'].select(range(1000))

import os
from ragstack_colbert import CassandraDatabase, ColbertEmbeddingModel

keyspace = "benchmarksmarco1000"
database_id = os.getenv("ASTRA_DATABASE_ID")
astra_token = os.getenv("ASTRA_TOKEN")

database = CassandraDatabase.from_astra(
    astra_token=astra_token,
    database_id=database_id,
    keyspace=keyspace
)

embedding_model = ColbertEmbeddingModel()

from ragstack_langchain.colbert import ColbertVectorStore as LangchainColbertVectorStore

lc_vector_store = LangchainColbertVectorStore(
    database=database,
    embedding_model=embedding_model,
)

all_texts = []
all_metadatas = []
i = 0
for row in subset:
    all_texts.extend(row['passages']['passage_text'])
    all_metadatas.extend([{'row_id': i} for _ in row['passages']['is_selected']])
    i += 1
