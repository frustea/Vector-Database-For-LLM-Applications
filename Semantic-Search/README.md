# 🌟 Vector Indexing and Querying Project

## 📌 Overview

This project demonstrates the integration of machine learning 🧠 and vector databases 🗃️ for efficient text processing and querying. It utilizes the SentenceTransformer library for generating sentence embeddings and Pinecone 🌲 for indexing and querying these embeddings in a scalable manner. The utility script provided streamlines the setup and management of Pinecone services.

## 🚀 Installation

To get started with this project, follow these steps:

1. **Clone the Repository** 📂

    ```
    git clone <repository-url>
    ```

2. **Install Dependencies** 📦

    Navigate to the project directory and install the required Python packages:

    ```
    pip install -r requirements.txt
    ```

## 📂 Files

- `vector_indexer.py`: Contains the `VectorIndexer` class responsible for creating vector embeddings and interacting with Pinecone's indexing and querying services.
- `utils.py`: Provides utility functions for configuring and managing Pinecone services.
- `main.py`: Demonstrates how to use the `VectorIndexer` class for indexing and querying operations.
- `requirements.txt`: Lists all the necessary Python packages to run the project.

## 🛠️ Usage

1. **Set up Pinecone** 🔑

    Before using the project, sign up for Pinecone and obtain an API key. Set up your environment with the Pinecone API key:

    ```
    export PINECONE_API_KEY='your_pinecone_api_key_here'
    ```

2. **Initialize the Indexer** 🌐

    Use the `main.py` script to initialize the `VectorIndexer`, create an index, and run queries:

    ```
    python main.py
    ```

    Ensure you modify `main.py` to include your specific dataset and queries.

3. **Customization** ⚙️

    You can customize the `VectorIndexer` and `utils.py` according to your project requirements, including changing the Pinecone index specifications or the SentenceTransformer model.

## 💡 Contributing

Contributions to this project are welcome. Please follow the standard GitHub fork 🍴 and pull request 📤 workflow.




