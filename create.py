from dotenv import load_dotenv, find_dotenv
import os
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm
import ast

_ = load_dotenv(find_dotenv())
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


pinecone = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "demotest"

if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
    pinecone.delete_index(INDEX_NAME)

pinecone.create_index(name=INDEX_NAME, dimension=1536, metric='cosine',
                      spec=ServerlessSpec(cloud='aws', region='us-east-1'))

index = pinecone.Index(INDEX_NAME)
print(index)

max_articles_num = 500
df = pd.read_csv('./wiki.csv')#, nrows=max_articles_num)
print(df.head())

prepped = []

for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    meta = ast.literal_eval(row['metadata'])
    prepped.append({'id':row['id'], 
                    'values':ast.literal_eval(row['values']), 
                    'metadata':meta})
    if len(prepped) >= 250:
        print("doing upsert")
        resp = index.upsert(prepped)
        prepped = []
        print(resp.upserted_count)

print(index.describe_index_stats())
