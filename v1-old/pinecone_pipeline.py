import os
import pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-3.5-turbo"

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-west1-gcp")


def get_vector_store(chunk_size: int):
    return PineconeVectorStore(
        embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        index_name=f"chunk_size_{chunk_size}"
    )


def ingest(file_path: str, chunk_size: int, **kwargs):
    vector_store = get_vector_store(chunk_size=chunk_size)

    chunk_overlap = min(chunk_size / 4, min(chunk_size / 2, 64))
    logging.info(f"Using chunk_overlap: {chunk_overlap} for chunk_size: {chunk_size}")

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=EMBEDDING_MODEL,
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
