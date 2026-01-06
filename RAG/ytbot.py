from langchain_community.vectorstores import Chroma ,FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.documents import Document
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
import os

token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

#Step 1a - Indexing (Document Ingestion)
loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=LPZh9BOjkQs",
    add_video_info=False,
    transcript_format=TranscriptFormat.CHUNKS,
    chunk_size_seconds=30
)
docs =loader.load()

transcript = " ".join(doc.page_content for doc in docs)

#Step 1b - Indexing (Text Splitting)
splitter =  RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=150)
chunks = splitter.create_documents([transcript])

#Step 1c & 1d - Indexing (Embedding Generation and Storing in Vector Store)
vector_store = FAISS.from_documents(chunks, embedding)
#print(vector_store.index_to_docstore_id)

#Step 2 - Retrieval
retriver = vector_store.as_retriever(search_type='similarity',search_kwargs={'k':2})

#Step 3 - Augmentation
load_dotenv()

token=os.getenv('HUGGINGFACEHUB_API_TOKEN')

llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task='text-generation',
    provider='hf-inference',
    huggingfacehub_api_token=token
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables=['context','question']
)
question  = "tell me about large language models that are discussed in this video"
retrieved_docs = retriver.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({'context':context_text,'question':question})

#Step 4 - Generation
answer = model.invoke(final_prompt)
print(answer.content)