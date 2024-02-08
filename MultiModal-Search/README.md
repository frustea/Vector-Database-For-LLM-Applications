# Hybrid Search Engine Project

Welcome to the Hybrid Search Engine project! This innovative solution combines the power of dense and sparse vector search to provide highly relevant search results for textual queries. Leveraging machine learning models and vector database technology, this project aims to enhance search capabilities across extensive datasets.

## Overview

The Hybrid Search Engine utilizes both BM25 (sparse vector) and Sentence Transformers (dense vector) to index and query datasets. This approach allows for a more nuanced search that considers semantic similarity and keyword matching, making it ideal for applications requiring high accuracy and relevancy in search results.

## Features

- Hybrid vector indexing using BM25 and Sentence Transformers.
- Efficient querying with adjustable weighting between dense and sparse vectors.
- Scalable indexing and querying designed for large datasets.

## Getting Started

### Prerequisites

- Python 3.6+
- Pinecone API key
- OpenAI API key (for Sentence Transformers)

### Installation

1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt


### Usage
1-Initialize the Hybrid Search Engine with your configuration:
```
from hybrid_search_engine import HybridSearchEngine

engine = HybridSearchEngine(
    index_name='your_index_name',
    dataset_name='your_dataset_name',
    dataset_split='train'
)

```

2-Index your dataset:
```
engine.encode_and_index_data()

```

3-Perform a hybrid query:
```
query = "your search query"
results = engine.hybrid_query(query, alpha=0.5, top_k=10)

```
4-Explore the results:
```
for result in results['matches']:
    print(result)

```
