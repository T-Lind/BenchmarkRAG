import os
import time

import weaviate
from dotenv import load_dotenv
from langchain_community.retrievers import (
    WeaviateHybridSearchRetriever, PineconeHybridSearchRetriever,
)
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
import openai


load_dotenv(override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings()

WEAVIATE_URL = os.getenv("WCS_URL")
auth_client_secret = (weaviate.AuthApiKey(api_key=os.getenv("WCS_API_KEY")),)
weaviate_client = weaviate.Client(
    url=WEAVIATE_URL,
    additional_headers={
        "X-Openai-Api-Key": os.getenv("OPENAI_API_KEY"),
    },
)

weaviate_retriever = WeaviateHybridSearchRetriever(
    client=weaviate_client,
    index_name="LangChain",
    text_key="text",
    attributes=[],
    create_schema_if_missing=True,
)
corpus = ["foo", "bar", "world", "hello"]


# load to your BM25Encoder object
bm25_encoder = BM25Encoder().load("bm25_values.json")

print(weaviate_retriever.invoke("The albatross is what?", score=True))

index_name = "langchain-pinecone-hybrid-search-1"

# initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


index = pc.Index(index_name)
retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings, sparse_encoder=bm25_encoder, index=index
)
result = retriever.search(corpus).invoke("The albatross is what?")
print(result[0])

