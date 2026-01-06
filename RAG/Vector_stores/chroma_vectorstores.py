from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Create LangChain documents for IPL players

import os 
token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

doc1 = Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"}
    )
doc2 = Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"}
    )
doc3 = Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"team": "Chennai Super Kings"}
    )
doc4 = Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians"}
    )
doc5 = Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"team": "Chennai Super Kings"}
    )

docs = [doc1,doc2,doc3,doc4,doc4]

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

vector_stores = Chroma(
    embedding_function = embedding,
    persist_directory = 'my_chroma_db',
    collection_name='sample'
)

#add docments to vector store
vector_stores.add_documents(docs)

#view documents
vector_stores.get(include=['embeddings','documents','metadatas'])

#search documents
vector_stores.similarity_search(
    query='who the players is a bowler?',
    k=2
)

#search with similarity scores
vector_stores.similarity_search_with_score(
    query='who the players is a bowler?',
    k=2
)

#search with metadata filtering
vector_stores.similarity_search_with_score(
    query='who the players is a bowler?',
    filter ={'team':'chennai super kings'}
)

#to update documents
#update documents
#updated_docs = Document(
 #   page_content='',
  #  metadata={}
#)
#vector_stores.update_documents(document_id='',document=updated_docs)

#to delete document
#delete documnets
#vector_stores.delete(ids=[])

