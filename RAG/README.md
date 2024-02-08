# 🌐 RAG Application: Transforming Queries into Informed Articles

Welcome to our RAG Application project repository! This innovative tool leverages the latest in machine learning and vector search technology to process extensive text datasets and respond to queries with highly relevant articles. 🚀📚

## 🎯 Main Idea

Our project harnesses the power of the Pinecone API to transform input datasets into vector datasets, enabling us to efficiently search and retrieve the most relevant entries and articles for any given query. We then take it a step further by utilizing the OpenAI API to generate new, insightful articles based on the most pertinent information extracted from the dataset. This process ensures that each response is not only relevant but also rich in content and tailored to the query's context. Whether you're looking for detailed explanations, summaries, or deep dives into specific topics, our RAG application has got you covered. It's perfect for educators, researchers, and anyone in need of comprehensive, contextually relevant information. 📖✨

## 🛠️ Installation

Get started by ensuring you have Python 3.6+ and pip installed. Our project depends on several key libraries, such as `pandas`, `torch`, `sentence-transformers`, `pinecone-client`, and more, which are all listed in our `requirements.txt`. Install all necessary dependencies with:
```
pip install -r requirements.txt
```

## 📁 Project Structure

- `DataVectorEmbedding.py`: Scripts for converting input datasets into vector datasets using the Pinecone API.
- `main.py`: The main script that orchestrates the query processing, vector search, and article generation workflow.
- `utils.py`: Contains utility functions for API key management and other configurations.
- `requirements.txt`: Specifies all project dependencies for easy setup.

## ⚙️ How to Use

1. Clone this repository to your local machine:

    ```
    git clone <repository-link>
    cd into-your-project-directory
    ```

2. Ensure you have the necessary API keys from Pinecone and OpenAI, and set them up as described in `utils.py`.

3. Run the `main.py` script to start the application:

    ```
    python main.py
    ```

4. Follow the on-screen prompts to enter your query and receive a comprehensive article tailored to your needs.



