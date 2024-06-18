# Some minor setup for the notebook
import warnings
import nest_asyncio
from tqdm import tqdm
import os
import openai
import os
import time
import openai
import weaviate
from dotenv import load_dotenv
from langchain_community.retrievers import (
    WeaviateHybridSearchRetriever,
)
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Ignore all warnings
warnings.filterwarnings("ignore")

# Allows for running async code in Jupyter notebooks
nest_asyncio.apply()
from langchain_core.documents import Document
from datasets import load_dataset
import os
from dotenv import load_dotenv
bioasq = load_dataset("rag-datasets/rag-mini-bioasq", "question-answer-passages")['test']
corpus = load_dataset("rag-datasets/rag-mini-bioasq", "text-corpus")['passages']
# Specify the number of results for each search.
n_results = 10
load_dotenv(override=True)
WEAVIATE_URL = os.getenv("WCS_URL")
WEAVIATE_API_KEY = os.getenv("WCS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY
auth_config = weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY)
client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=auth_config,
    additional_headers={
        "X-Openai-Api-Key": OPENAI_API_KEY,
    },
)
embeddings = OpenAIEmbeddings()
weaviate_retriever = WeaviateHybridSearchRetriever(
    client=client,
    index_name="LangChain",
    text_key="text",
    attributes=['index'],
    create_schema_if_missing=True,
    k=n_results,
)
# ONLY RUN ONCE!
import time
from tqdm import tqdm
corpus_docs = []
str_time = time.strftime("%m/%d/%Y")
for doc in tqdm(corpus):
    corpus_docs.append(Document(
        metadata={
            "name": "rag-mini-bioasq",
            "date": str_time,
            "index": doc['id'],
        },
        page_content=doc['passage']
    ))
weaviate_retriever.add_documents(corpus_docs)
