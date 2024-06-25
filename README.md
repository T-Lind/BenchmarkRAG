# Benchmark RAG
Author: Tiernan Lindauer, Datastax Inc.

## Introduction

This repository tests AstraDB's ColBERT vector search implementation compared to Pinecone and Weaviate's Hybrid Search implementations. It tests latency, cost, and relevancy.

The new version is a WIP, testing ColBERT vs MRR using the MS MARCO dataset. ColBERT scores a 0.9825, and BM25 scores a 0.7442. 

## [OLD] v1 Introduction
This repository tests AstraDB's ColBERT vector search implementation compared to Pinecone and Weaviate's Hybrid Search implementations. It tests latency, cost, and relevancy.

The main file is `v1-old/benchmark.py` which runs the tests. You'll need to have Jupiter Notebook installed to run the tests.

This repository and evaluation is open-source, and does not contain any proprietary information.
