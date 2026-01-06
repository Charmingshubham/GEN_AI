from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
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

prompt1 = template1.invoke({"topic": "Climate Change"})
result1 = model.invoke(prompt1)

prompt2 = template2.invoke({"report": result1})
result2 = model.invoke(prompt2)

print(result2.content)