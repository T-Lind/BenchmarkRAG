from datasets import load_dataset
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
from ragstack_colbert import CassandraDatabase, ColbertEmbeddingModel
from ragstack_langchain.colbert import ColbertVectorStore as LangchainColbertVectorStore

load_dotenv()

ms_marco = load_dataset("microsoft/ms_marco", "v2.1")
subset = ms_marco['train']  # Do all of it

keyspace = "benchmarksmarcoall"
database_id = os.getenv("ASTRA_DATABASE_ID")
astra_token = os.getenv("ASTRA_TOKEN")

database = CassandraDatabase.from_astra(
    astra_token=astra_token,
    database_id=database_id,
    keyspace=keyspace
)

embedding_model = ColbertEmbeddingModel()

lc_vector_store = LangchainColbertVectorStore(
    database=database,
    embedding_model=embedding_model,
)

all_texts = []
all_metadatas = []
I = 0
for row in subset:
    all_texts.extend(row['passages']['passage_text'])
    all_metadatas.extend([{'row_id': I} for _ in row['passages']['is_selected']])
    I += 1


def add_texts_batch(start_idx):
    batch_texts = all_texts[start_idx:start_idx + 20]
    batch_metadatas = all_metadatas[start_idx:start_idx + 20]
    lc_vector_store.add_texts(batch_texts, metadatas=batch_metadatas)


batch_size = 20
num_batches = len(all_texts) // batch_size

with ThreadPoolExecutor(max_workers=1000) as executor:
    futures = [executor.submit(add_texts_batch, i * batch_size) for i in range(num_batches + 1)]
    for future in tqdm(as_completed(futures)):
        future.result()
