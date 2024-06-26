import os
from pinecone import Pinecone, ServerlessSpec
from langchain_community.retrievers import (
    PineconeHybridSearchRetriever,
)
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv(override=True)

ms_marco = load_dataset("microsoft/ms_marco", "v2.1")

# Use subset for demonstration
subset = ms_marco['train'].select(range(1000))

index_name = "benchmarksmarco1000parallel"

all_texts = []
all_metadatas = []
i = 0
for row in subset:
    all_texts.extend(row['passages']['passage_text'])
    all_metadatas.extend([{'row_id': i} for _ in row['passages']['is_selected']])
    i += 1

# initialize Pinecone client
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index = pc.Index(index_name)

from langchain_openai import OpenAIEmbeddings

openai_embeddings = OpenAIEmbeddings()

from pinecone_text.sparse import BM25Encoder

# or from pinecone_text.sparse import SpladeEncoder if you wish to work with SPLADE

# use default tf-idf values
bm25_encoder = BM25Encoder().default()

bm25_encoder.fit(all_texts)

cwd = os.getcwd()

bm25_encoder.dump(cwd + "/bm25_encoder.json")

retriever = PineconeHybridSearchRetriever(
    embeddings=openai_embeddings, sparse_encoder=bm25_encoder, index=index
)


def add_texts_batch(start_idx):
    batch_texts = all_texts[start_idx:start_idx + 20]
    batch_metadatas = all_metadatas[start_idx:start_idx + 20]
    retriever.add_texts(batch_texts, metadatas=batch_metadatas)


batch_size = 20
num_batches = len(all_texts) // batch_size

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

with ThreadPoolExecutor(max_workers=100) as executor:
    futures = [executor.submit(add_texts_batch, i * batch_size) for i in range(num_batches + 1)]
    for future in tqdm(as_completed(futures)):
        future.result()
