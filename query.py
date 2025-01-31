import ast
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import os
import pandas as pd
from pinecone import Pinecone

_ = load_dotenv(find_dotenv())
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#def get_embeddings(articles, model="text-embedding-ada-002"):
#   return openai_client.embeddings.create(input = articles, model=model)

def get_embeddings(articles):
    df = pd.read_csv("embed.csv")
    df['embedding'] = df['embedding'].apply(ast.literal_eval)
    return df['embedding'].iloc[0]


pinecone = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "demotest"

index = pinecone.Index(INDEX_NAME)

print(index.describe_index_stats())

query = "what is the berlin wall?"
embed = get_embeddings([query])
res = index.query(vector=embed, top_k=3, include_metadata=True)
text = [r['metadata']['text'] for r in res['matches']]
print('\n'.join(text))


### RAG
query = "write an article titled: what is the berlin wall?"
embed = get_embeddings([query])
res = index.query(vector=embed, top_k=3, include_metadata=True)

contexts = [
    x['metadata']['text'] for x in res['matches']
]

prompt_start = (
    "Answer the question based on the context below.\n\n"+
    "Context:\n"
)

prompt_end = (
    f"\n\nQuestion: {query}\nAnswer:"
)

prompt = (
    prompt_start + "\n\n---\n\n".join(contexts) + 
    prompt_end
)

print(prompt)

openai_client = OpenAI(api_key=OPENAI_API_KEY)
res = openai_client.completions.create(
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
