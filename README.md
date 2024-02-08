# Vector-Database-For-LLM-Applications
Exploring the Application of Vector Databases (VD) in the LLM, including Semantic Search and RAG

## Subprojects Overview

### 🌟 Vector Indexing and Querying Project

This subproject focuses on efficiently processing and querying text data. It leverages the SentenceTransformer library for generating sentence embeddings and the Pinecone service for indexing and querying these embeddings. Ideal for applications requiring fast and scalable text search capabilities.

- [Vector Indexing and Querying Project README](./Semantic-Search/README.md)

### 🌐 RAG Application: Transforming Queries into Informed Articles

The RAG (Retrieval-Augmented Generation) Application transforms queries into comprehensive articles by leveraging vector search technology and machine learning. It utilizes Pinecone for vector dataset transformation and OpenAI API for generating informed articles based on query-relevant entries.

- [RAG Application README](./RAGA/README.md)

### Hybrid Search Engine Project

Combining dense and sparse vector search, the Hybrid Search Engine offers nuanced search capabilities that balance semantic similarity and keyword matching. It uses BM25 for sparse vector indexing and Sentence Transformers for dense vector indexing, providing a scalable solution for high-accuracy search results.

- [Hybrid Search Engine Project README](./Multimodal/README.md)

## Getting Started

To explore these subprojects, clone this repository and navigate to each subproject directory:

```bash
git clone <repository-url>
cd <subproject-directory>
