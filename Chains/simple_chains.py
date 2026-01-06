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

template = PromptTemplate(
    template='summary about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

chain = template | model | parser

result = chain.invoke({'topic':'llm'})

print(result)

#to visualise the pipeline
#chain.get_graph().print_ascii()