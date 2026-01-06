from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

token = os.getenv('HUGGING_FACE_API_KEY')

llm = HuggingFaceEndpoint(
    repo_id = 'mistralai/Mistral-7B-Instruct-v0.2',
    task = 'text-generation',
    provider= 'auto',
    huggingfacehub_api_token=token
)

model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="what is the capital of india")
]

result =model.invoke(messages)

messages.append(AIMessage(content = result.content))

print(messages)