from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

token=os.getenv('HUGGINGFACEHUB_API_TOKEN')

llm = HuggingFaceEndpoint(
    repo_id='moonshotai/Kimi-K2-Thinking',
    task='text-generation',
    provider='auto',
    huggingfacehub_api_token=token
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='write the summary of {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

loader = TextLoader('documents.txt')

docs=loader.load()

#print(len(docs))

#print(docs[0].metadata)

#print(docs[0].page_content)

chain = prompt | model | parser

print(chain.invoke({'topic':docs[0].page_content}))
