from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
import os 

load_dotenv()

token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

llm = HuggingFaceEndpoint(
    repo_id="moonshotai/Kimi-K2-Thinking",
    temperature=0.7,
    provider='auto',
    huggingfacehub_api_token=token)

model = ChatHuggingFace(llm=llm)

#1st prompt: detailed report
template1 = PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=["topic"]
)

#2nd prompt: summarize the detailed report
template2 = PromptTemplate(
    template="summarize the following report in a concise manner: {report}",
    input_variables=["report"]
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result=chain.invoke({'topic' : 'Climate Change'})
print(result)