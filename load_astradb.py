import os
import csv
import time
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ragstack_colbert import CassandraDatabase, ColbertEmbeddingModel, ColbertVectorStore

# Load environment variables
load_dotenv()

astra_database_id = os.getenv('ASTRA_DATABASE_ID')
astra_token = os.getenv('ASTRA_TOKEN')
keyspace = "benchmarking"

# Initialize AstraDB
database = CassandraDatabase.from_astra(astra_token=astra_token, database_id=astra_database_id, keyspace=keyspace)
embedding_model = ColbertEmbeddingModel()
vector_store = ColbertVectorStore(database=database, embedding_model=embedding_model)


# Function to load text files from a folder
def load_text_files(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                try:
                    texts.append(file.read())
                except UnicodeDecodeError:
                    print("Couldn't read file {}".format(filename))
    return texts


# Function to split text into chunks
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    return chunks


# Function to get embeddings using ColBERT
def get_embeddings(chunks):
    embeddings = []
    embedding = embedding_model.embed_texts(chunks)
    embeddings.append(embedding)
    return embeddings


# Function to save chunks and embeddings to CSV
def save_to_csv(chunks, embeddings, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['chunk', 'embedding'])
        for chunk, embedding in zip(chunks, embeddings):
            writer.writerow([chunk, embedding])


# Function to store chunks and embeddings in AstraDB
def store_in_astra(chunks):
    metadatas = []
    for i in range(len(chunks)):
        metadatas.append({'run': 'test-2', 'chunk': chunks[i]})
    vector_store.add_texts(chunks, metadatas=metadatas)


# Main function
def process_texts(folder_path, output_file):
    start_time = time.time()  # Start timer
    texts = load_text_files(folder_path)
    all_chunks = []
    # all_embeddings = []
    for text in texts:
        print("Processing \"{}\"".format(text[:35]))
        chunks = split_text(text)
        # embeddings = get_embeddings(chunks)
        all_chunks.extend(chunks)
        # all_embeddings.extend(embeddings)
    # save_to_csv(all_chunks, all_embeddings, output_file)
    store_in_astra(all_chunks)
    end_time = time.time()  # End timer
    print(f"Processing completed in {end_time - start_time:.2f} seconds")


# Run the process
process_texts('dataset/movie_scripts/test', 'output.csv')
