from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline
from dotenv import load_dotenv
import os

load_dotenv()

os.enviorn['HF_HOME'] = 'D:/miniproject/hf_cache'

token = os.getenv('HUGGINGFACEHUB_ACCESS_TOKEN')

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task = 'text-generation',
    huggingfacehub_api_token = token,
    pipeline_kwargs={'max_new_tokens':200,'temperature':0.3}

)

model = ChatHuggingFace(llm=llm)

result = model.invoke('What is the capital of Spain?')
print(result.content)