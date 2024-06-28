# Benchmark RAG
Author: Tiernan Lindauer, Datastax Inc.

## Introduction

This repository tests AstraDB's ColBERT vector search implementation compared to Pinecone and Weaviate's Hybrid Search implementations. It tests latency, cost, and relevancy.

To get started, set up your environment and run
```bash
pip install -r requirements.txt
```

You will need to setup a `.env` file, with the following variables:
```env
OPENAI_API_KEY=sk-YOUROPENAIAPIKEY
ASTRA_DATABASE_ID=YOUR-ASTRA-DATABASE-ID
ASTRA_API_ENDPOINT=https://YOURASTRAAPIENDPOINT-REGION.apps.astra.datastax.com
ASTRA_TOKEN=AstraCS:ASTRATOKEN
PINECONE_API_KEY=PINECONE-API-KEY
PINECONE_INDEX_NAME=YOURPINECONEINDEXNAME
WCS_URL=https://YOURWEAVIATECLUSTERURL.weaviate.network
WCS_API_KEY=YOUR_WEAVIATE_API_KEY
```

Next, ensure Jupyter is set up on your system, and run `benchmark.ipynb` to run the benchmarking. The end of the notebook has a series of tests to verify that the results are correct.



## [OLD] v1 Introduction
This repository tests AstraDB's ColBERT vector search implementation compared to Pinecone and Weaviate's Hybrid Search implementations. It tests latency, cost, and relevancy.

The main file is `v1-old/benchmark.py` which runs the tests. You'll need to have Jupiter Notebook installed to run the tests.

This repository and evaluation is open-source, and does not contain any proprietary information.
