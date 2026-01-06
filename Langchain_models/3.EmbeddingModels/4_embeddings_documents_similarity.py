from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()   

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',dimensions=200)

docs = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
    "The capital of Italy is Rome.",
    "The capital of Spain is Madrid."
]

query = "What is the capital of Germany?"

doc_vector = embeddings.embed_documents(docs)
query_vector = embeddings.embed_query(query)

scores = cosine_similarity([query_vector],doc_vector)[0]

print(sorted(list(enumerate(scores)),key=lambda x:x[1],reverse=True))