from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
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

parser = JsonOutputParser()

template = PromptTemplate(
    template = 'give me the name of a country and its capital city and its population \n {format_instructions}',
    input_variables=[],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({})

print(result)

#the flaw if the json outptut parser is that ew cannot define schema here, means in what type of format we want the output
