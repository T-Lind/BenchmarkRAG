import os
import weaviate
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import (
    WeaviateHybridSearchRetriever,
)
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_core.documents import Document

load_dotenv(override=True)

ms_marco = load_dataset("microsoft/ms_marco", "v2.1")

# Use subset for demonstration
subset = ms_marco['train'].select(range(1000))

auth_config = weaviate.AuthApiKey(api_key=os.environ['WCS_API_KEY'])
client = weaviate.Client(
    url=os.environ['WCS_URL'],
    auth_client_secret=auth_config,
    additional_headers={
        "X-Openai-Api-Key": os.environ["OPENAI_API_KEY"],
    },
)
embeddings = OpenAIEmbeddings()

weaviate_retriever = WeaviateHybridSearchRetriever(
    client=client,
    index_name="LangChain",
    text_key="text",
    attributes=['index'],
    create_schema_if_missing=True,
)

all_texts = []
all_metadatas = []
i = 0
for row in subset:
    all_texts.extend(row['passages']['passage_text'])
    all_metadatas.extend([{'row_id': i} for _ in row['passages']['is_selected']])
    i += 1

docs = []
for text, metadata in zip(all_texts, all_metadatas):
    docs.append(Document(
        metadata=metadata,
        page_content=text
    ))

weaviate_retriever.add_documents(docs)
