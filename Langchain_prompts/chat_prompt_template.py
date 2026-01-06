from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace 
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

token = os.getenv('HUGGING_FACE_API_KEY')

llm = HuggingFaceEndpoint(
    repo_id = 'mistralai/Mistral-7B-Instruct-v0.2',
    task = 'text-generation',
    huggingfacehub_api_token=token,
    provider= 'auto'
)  

model = ChatHuggingFace(llm=llm)

chat_template = ChatPromptTemplate(
    [('system','you are a helpful {domain} assistant'),
     ('human','explain in simple terms what is {topic}')]
)

prompt = chat_template.invoke({'domain':'science','topic':'quantum computing'})

print(prompt)