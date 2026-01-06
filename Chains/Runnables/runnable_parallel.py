from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv  
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel ,RunnableSequence
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

prompt = PromptTemplate(
    template='generate a tweet about {topic}',
    input_variables=['topic']
)
prompt1= PromptTemplate(
    template='generate a linkdin post about {topic}',
    input_variables=['topic']
)

chain = RunnableParallel({
    'tweet' : RunnableSequence(prompt,model,parser),
    'linkdin' : RunnableSequence(prompt1,model,parser)
})

result = chain.invoke({'topic':'ai'})

print(result['tweet'])
print(result['linkdin'])