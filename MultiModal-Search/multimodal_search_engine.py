import torch
from datasets import load_dataset
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from pinecone_text.sparse import BM25Encoder
from tqdm.auto import tqdm
from DLAIUtils import Utils
import pandas as pd

class HybridSearchEngine:
    def __init__(self, index_name, dataset_name, dataset_split, batch_size=100, fashion_data_num=1000, dimension=512, metric="dotproduct", cloud='aws', region='us-west-2'):
        self.utils = Utils()
        self.pinecone_api_key = self.utils.get_pinecone_api_key()
        Pinecone.init(api_key=self.pinecone_api_key)
        self.index_name = self.utils.create_dlai_index_name(index_name)
        self.dimension = dimension
        self.metric = metric
        self.cloud = cloud
        self.region = region
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.batch_size = batch_size
        self.fashion_data_num = fashion_data_num
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.initialize_index()
        self.bm25 = BM25Encoder()
        self.model = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)

    def initialize_index(self):
        if self.index_name in [index.name for index in Pinecone.list_indexes()]:
            Pinecone.delete_index(self.index_name)
        Pinecone.create_index(
            name=self.index_name,
            dimension=self.dimension,
            metric=self.metric,
            spec=ServerlessSpec(cloud=self.cloud, region=self.region)
        )
        self.index = Pinecone.Index(self.index_name)

    def encode_and_index_data(self):
        fashion = load_dataset(self.dataset_name, split=self.dataset_split)
        metadata = fashion.to_pandas()
        self.bm25.fit(metadata['productDisplayName'])
        for i in tqdm(range(0, min(self.fashion_data_num, len(fashion)), self.batch_size)):
            i_end = min(i+self.batch_size, len(fashion))
            meta_batch = metadata.iloc[i:i_end]
            meta_dict = meta_batch.to_dict(orient="records")
            meta_batch = [" ".join(x) for x in meta_batch.loc[:, ~meta_batch.columns.isin(['id', 'year'])].values.tolist()]
            sparse_embeds = self.bm25.encode_documents(meta_batch)
            dense_embeds = self.model.encode(meta_batch).tolist()
            ids = [str(x) for x in range(i, i_end)]
            upserts = [{
                'id': _id,
                'sparse_values': sparse,
                'values': dense,
                'metadata': meta
            } for _id, sparse, dense, meta in zip(ids, sparse_embeds, dense_embeds, meta_dict)]
            self.index.upsert(upserts)

    def hybrid_query(self, query, alpha=0.5, top_k=10):
        sparse = self.bm25.encode_queries(query)
        dense = self.model.encode([query]).tolist()[0]
        hdense, hsparse = self.hybrid_scale(dense, sparse, alpha)
        result = self.index.query(top_k=top_k, vector=hdense, sparse_vector=hsparse, include_metadata=True)
        return result

    def hybrid_scale(self, dense, sparse, alpha):
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")
        hsparse = {'indices': sparse['indices'], 'values': [v * (1 - alpha) for v in sparse['values']]}
        hdense = [v * alpha for v in dense]
        return hdense, hsparse
