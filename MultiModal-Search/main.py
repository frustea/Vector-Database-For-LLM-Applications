from multimodal_search_engine import HybridSearchEngine

def main():
    # Initialize the Hybrid Search Engine with specific parameters
    engine = HybridSearchEngine(
        index_name='dl-ai',
        dataset_name='ashraq/fashion-product-images-small',
        dataset_split='train',
        batch_size=100,
        fashion_data_num=1000,
        dimension=512,
        metric="dotproduct",
        cloud='aws',
        region='us-west-2'
    )
    
    # Encode and index the dataset
    print("Encoding and indexing dataset...")
    engine.encode_and_index_data()
    print("Indexing complete.")
    
    # Perform a hybrid query
    query = "dark blue french connection jeans for men"
    print(f"Performing hybrid query for: '{query}'")
    results = engine.hybrid_query(query, alpha=0.5, top_k=10)
    
    # Display the results
    print("Query Results:")
    for result in results['matches']:
        print(f"ID: {result['id']}, Metadata: {result['metadata']}")

if __name__ == "__main__":
    main()
