import os
from rank_bm25 import BM25Okapi
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

LLM_MODEL = "gpt-3.5-turbo"


# BM25 Vector Store
class BM25VectorStore:
    def __init__(self):
        self.docs = []
        self.bm25 = None

    def add_documents(self, documents):
        self.docs.extend(documents)
        tokenized_corpus = [doc.content.split(" ") for doc in self.docs]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def as_retriever(self, search_kwargs):
        return BM25Retriever(self.bm25, self.docs, search_kwargs["k"])


class BM25Retriever:
    def __init__(self, bm25, docs, k):
        self.bm25 = bm25
        self.docs = docs
        self.k = k

    def __call__(self, query):
        tokenized_query = query.split(" ")
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:self.k]
        return [self.docs[i] for i in top_k_indices]


def get_vector_store(chunk_size: int):
    return BM25VectorStore()


def ingest(file_path: str, chunk_size: int, **kwargs):
    vector_store = get_vector_store(chunk_size=chunk_size)

    chunk_overlap = min(chunk_size / 4, min(chunk_size / 2, 64))
    logging.info(f"Using chunk_overlap: {chunk_overlap} for chunk_size: {chunk_size}")

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=50,
    )

    docs = UnstructuredFileLoader(
        file_path=file_path, mode="single", strategy="fast"
    ).load()
    split_docs = text_splitter.split_documents(docs)
    vector_store.add_documents(split_docs)


def query_pipeline(k: int, chunk_size: int, **kwargs):
    vector_store = get_vector_store(chunk_size=chunk_size)
    llm = ChatOpenAI(model_name=LLM_MODEL)

    # build a prompt
    prompt_template = """
    Answer the question based only on the supplied context. If you don't know the answer, say: "I don't know".
    Context: {context}
    Question: {question}
    Your answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    rag_chain = (
            {
                "context": vector_store.as_retriever(search_kwargs={"k": k}),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain
