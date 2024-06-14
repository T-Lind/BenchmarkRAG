import os
from dotenv import load_dotenv
import numpy as np
from sklearn.neighbors import NearestNeighbors
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv(override=True)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()


def load_and_split_text(folder_path):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            text = file.read()
            texts.extend(text_splitter.split_text(text))
    return texts


def embed_texts(texts):
    return np.array(embeddings.embed_documents(texts))


def embed_queries(queries):
    return np.array(embeddings.embed_documents(queries))


def load_queries(query_file):
    with open(query_file, 'r') as file:
        queries = file.readlines()
    return [query.strip() for query in queries]


def save_results(results_file, results):
    with open(results_file, 'w') as file:
        for result in results:
            file.write(" ".join(map(str, result)) + "\n")


def main(corpus_folder, query_file, results_file):
    # Load and split corpus
    corpus_texts = load_and_split_text(corpus_folder)
    corpus_embeddings = embed_texts(corpus_texts)

    # Load and embed queries
    queries = load_queries(query_file)
    query_embeddings = embed_queries(queries)

    # Use KNN to find the closest 5 documents
    knn = NearestNeighbors(n_neighbors=5, metric='cosine').fit(corpus_embeddings)

    results = []
    for query_embedding in query_embeddings:
        distances, indices = knn.kneighbors([query_embedding])
        results.append(indices[0])

    # Save results to file
    save_results(results_file, results)


if __name__ == "__main__":
    corpus_folder = "dataset/movie_scripts/test"
    query_file = "dataset/queries/holy_grail.txt"
    main(corpus_folder, query_file, "dataset/queries/holy_grail_result.txt")
