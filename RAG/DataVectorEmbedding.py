from datasets import load_dataset
import pandas as pd
import ast
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from utils import Utils
import openai
from openai import OpenAI

class ArticleProcessor:
    def __init__(self, df,max_articles_num=500):
        self.utils = Utils()
        self.PINECONE_API_KEY = self.utils.get_pinecone_api_key()
        #self.df = pd.read_csv('./data/wiki.csv', nrows=max_articles_num)
        self.df=df
        self.index = self.initialize_pinecone_index()
        
        
        self.OPENAI_API_KEY = self.utils.get_openai_api_key()
        self.openai_client = OpenAI(api_key=self.OPENAI_API_KEY)

    def initialize_pinecone_index(self):
        pinecone = Pinecone(api_key=self.PINECONE_API_KEY)
        index_name = self.utils.create_dlai_index_name('dl-ai')
        
        if index_name in [index.name for index in pinecone.list_indexes()]:
            pinecone.delete_index(index_name)
        
        pinecone.create_index(name=index_name, dimension=1536, metric='cosine',
                              spec=ServerlessSpec(cloud='aws', region='us-west-2'))
        
        return pinecone.Index(index_name)
  

    def vectorize_dataset(self):
        prepped = []
        for i, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            meta = ast.literal_eval(row['metadata'])
            prepped.append({'id': row['id'], 'values': ast.literal_eval(row['values']), 'metadata': meta})
            if len(prepped) >= 250:
                self.index.upsert(prepped)
                prepped = []

    #@staticmethod
    def get_embeddings(self,articles, model="text-embedding-ada-002"):
        
        
        return self.openai_client.embeddings.create(input = articles, model=model)

    def build_prompt(self, query):
        embed = self.get_embeddings([query])
        
        res = self.index.query(vector=embed.data[0].embedding, top_k=3, include_metadata=True)

        contexts = [x['metadata']['text'] for x in res['matches']]
        prompt_start = "Answer the question based on the context below.\n\nContext:\n"
        prompt_end = f"\n\nQuestion: {query}\nAnswer:"
        prompt = prompt_start + "\n\n---\n\n".join(contexts) + prompt_end

        return prompt

    def generate_completion(self, prompt):
        res = self.openai_client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        temperature=0,
        max_tokens=636,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
        )
        print('-' * 80)
        print(res.choices[0].text)
        
       

        return res.choices[0].text

