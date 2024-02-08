
from vector_indexer import *

# defining 'dataset' 
dataset = load_dataset('quora', split='train[240000:290000]')
# defining query 
query="How to take a darkframe with camera?"


vector_indexer = VectorIndexer()
vector_indexer.initialize_index()
vector_indexer.convert_dataset_to_vectors(dataset, batch_size=10, vector_limit=100)
vector_indexer.run_query(query, top_k=5)
