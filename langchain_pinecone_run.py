from langchain_openai import OpenAIEmbeddings
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import nltk

nltk.data.path.append('/Users/tiernan.lindauer/nltk_data')

load_dotenv(override=True)

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


index_name = "langchain-pinecone-hybrid-search-3"

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create the index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Dimensionality of dense model
        metric="dotproduct",  # Sparse values supported only for dotproduct
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

corpus = load_and_split_text("dataset/movie_scripts/test")

# Use default tf-idf values
bm25_encoder = BM25Encoder().default()
# Fit tf-idf values on your corpus
bm25_encoder.fit(corpus)

# Store the values to a json file
bm25_encoder.dump("bm25_values.json")

# Load to your BM25Encoder object
bm25_encoder = BM25Encoder().load("bm25_values.json")

retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings, sparse_encoder=bm25_encoder, index=index
)

retriever.add_texts(corpus)
result = retriever.invoke("What type of Albatross was it?")
print(result[0])
