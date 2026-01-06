from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace  
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

chat_template = ChatPromptTemplate([
    ('system','you are a helpful agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','query')
])

chat_history = []
#load chat history from file
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

prompt = chat_template.invoke({'chat_history':chat_history,'query':'what is the update on my refund?'})

print([prompt])