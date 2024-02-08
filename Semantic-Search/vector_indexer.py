import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import pinecone
from pinecone import Pinecone, ServerlessSpec
from utils import Utils
from datasets import load_dataset

class VectorIndexer:
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        """
        Initializes the VectorIndexer with a model and device setup.

        Parameters:
            model_name (str): Identifier for the SentenceTransformer model.
            device (str, optional): The computing device ('cuda' or 'cpu'). If None, automatically determined.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if self.device != 'cuda':
            print('CUDA is not available. Using CPU instead.')
        self.model = SentenceTransformer(model_name, device=self.device)
        self.index = None  # Pinecone index will be initialized later

    def initialize_index(self):
        """
        Initializes and configures the Pinecone index based on the model's embedding dimension.
        """
        utils = Utils()
        PINECONE_API_KEY = utils.get_pinecone_api_key()
        pinecone = Pinecone(api_key=PINECONE_API_KEY)
        INDEX_NAME = utils.create_dlai_index_name('dl-ai')

        if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
            pinecone.delete_index(INDEX_NAME)
        print(INDEX_NAME)
        pinecone.create_index(name=INDEX_NAME, 
            dimension=self.model.get_sentence_embedding_dimension(), 
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-west-2'))

        self.index = pinecone.Index(INDEX_NAME)
        
        

    def convert_dataset_to_vectors(self, dataset, batch_size, vector_limit):
        """
        Processes and uploads a dataset to the Pinecone index in vector format.
        
        Parameters:
            dataset (Dataset): The dataset to process.
            batch_size (int): The number of records per batch.
            vector_limit (int): The maximum number of vectors to process.
        """
        questions = []
        for record in dataset['questions']:
            questions.extend(record['text'])
        questions = list(set(questions))
        
        
        questions = questions[:vector_limit]

        for i in tqdm(range(0, len(questions), batch_size), desc="Processing batches"):
            i_end = min(i + batch_size, len(questions))
            ids = [str(x) for x in range(i, i_end)]
            metadatas = [{'text': text} for text in questions[i:i_end]]
            embeddings = self.model.encode(questions[i:i_end])
            records = zip(ids, embeddings, metadatas)
            self.index.upsert(vectors=records)

    def run_query(self, query, top_k):
        """
        Runs a query against the Pinecone index and prints the top_k results.
        
        Parameters:
            query (str): The query string.
            top_k (int): The number of top results to return.
        """
        embedding = self.model.encode(query).tolist()
        results = self.index.query(top_k=top_k, vector=embedding, include_metadata=True, include_values=False)
        for result in results['matches']:
            print(f"{round(result['score'], 2)}: {result['metadata']['text']}")
