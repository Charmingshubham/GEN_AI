from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv  
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

llm = HuggingFaceEndpoint(
    repo_id = 'MiniMaxAI/MiniMax-M2',
    task = 'text=generation',
    provider='auto',
    huggingfacehub_api_token=token
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

template1 = PromptTemplate(
    template='give a decribed report on {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='extract 5 important points from {text}',
    input_variables=['text']
) 

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'tesla car'})

print(result)