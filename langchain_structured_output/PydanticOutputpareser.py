from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field  
from dotenv import load_dotenv
import os 

load_dotenv()

token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

llm = HuggingFaceEndpoint(
    repo_id="moonshotai/Kimi-K2-Thinking",
    temperature=0.7,
    provider='auto',
    huggingfacehub_api_token=token)

model = ChatHuggingFace(llm=llm)

class student_info(BaseModel):
    name: str = Field(description="The full name of the student")
    age: int = Field(gt=18,description="The age of the student in years")
    city: str = Field(description="The city where the student lives")

parser = PydanticOutputParser(pydantic_object=student_info)

template = PromptTemplate(
    template = 'generate a student profile with name , age, city of {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)


chain = template | model | parser

result = chain.invoke({'place' : 'new york'})

print(result)