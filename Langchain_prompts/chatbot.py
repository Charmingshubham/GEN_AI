from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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

chat_history = [
    SystemMessage(content="you are a helpfull assitant")
]

while True:
    user_input = input('you:')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print('AI:',result.content)

print(chat_history)