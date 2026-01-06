from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv  
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser 
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnableBranch
from pydantic import BaseModel, Field
from typing import Literal
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



class Feedback(BaseModel):
    sentiment: Literal['positive','negative'] = Field(description='classify the sentiment in postive or negative')

parser = PydanticOutputParser(pydantic_object=Feedback)

parser2 = StrOutputParser()

prompt = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

prompt1 = PromptTemplate(
    template='write an appropriate response on this positive feedback {feedback}',
    input_variables=['feedback']
)

prompt2 = PromptTemplate(
    template='write an appropriate response on this negative feedback {feedback}',
    input_variables=['feedback']
)

classifier_chain = prompt | model | parser

branch_chain = RunnableBranch(
    (lambda x:x.sentiment=='positive',prompt1 | model | parser2),
    (lambda x:x.sentiment=='negative',prompt2 | model | parser2),
    RunnableLambda(lambda x : "could not find sentiment")
)

final_chain = classifier_chain | branch_chain

result = final_chain.invoke({'feedback':'this a terible smartphone'})

print(result)