from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatOpenAI(mdoel='gpt-4',temperature=0.2)

result = chat_model.invoke('What is the capital of Germany?')

print(result.content)