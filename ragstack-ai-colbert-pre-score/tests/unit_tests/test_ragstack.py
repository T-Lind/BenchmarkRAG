def test_import():
    from llama_index.vector_stores.astra_db import AstraDBVectorStore  # noqa
    from llama_index.vector_stores.cassandra import CassandraVectorStore  # noqa
    from langchain.vectorstores import AstraDB  # noqa
    from langchain_astradb import AstraDBVectorStore  # noqa
    import langchain_core  # noqa
    import langsmith  # noqa
    import astrapy  # noqa
    import cassio  # noqa
    import unstructured  # noqa
    import openai  # noqa
    import tiktoken  # noqa


def test_meta():
    from importlib import metadata

    def check_meta(package: str):
        meta = metadata.metadata(package)
        assert meta["version"]
        assert meta["license"] == "BUSL-1.1"

    check_meta("ragstack-ai")
    check_meta("ragstack-ai-langchain")
    check_meta("ragstack-ai-llamaindex")
    check_meta("ragstack-ai-colbert")


