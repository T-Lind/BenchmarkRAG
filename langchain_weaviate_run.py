import os
import time

import weaviate
from dotenv import load_dotenv
from langchain_community.retrievers import (
    WeaviateHybridSearchRetriever,
)
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True)

WEAVIATE_URL = os.getenv("WCS_URL")
client = weaviate.Client(
    url=WEAVIATE_URL,
    additional_headers={
        "X-Openai-Api-Key": os.getenv("OPENAI_API_KEY"),
    },
)

retriever = WeaviateHybridSearchRetriever(
    client=client,
    index_name="LangChain",
    text_key="text",
    attributes=[],
    create_schema_if_missing=True,
)

embeddings = OpenAIEmbeddings()


def load_and_split_text(folder_path):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            text = file.read()
            texts.extend(text_splitter.split_text(text))
    return texts


corpus = load_and_split_text("dataset/movie_scripts/test")

docs = []
index = 0
for doc in corpus:
    docs.append(Document(
        metadata={
            "name": "montypython",
            "date": time.strftime("%m/%d/%Y"),
            "index": index,
        },
        page_content=doc
    ))
    index += 1

retriever.add_documents(docs)
print(retriever.invoke("Who is the director of \"Monty Python and the Holy Grail\"?", score=True))
