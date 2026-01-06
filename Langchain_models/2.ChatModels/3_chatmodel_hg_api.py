from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-8B",
    task = "text-generation",
    huggingfacehub_api_token = token,
    provider="auto"
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of Spain?")
print(result.content)