from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv  
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
import os

load_dotenv()

token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

llm1 = HuggingFaceEndpoint(
    repo_id = 'MiniMaxAI/MiniMax-M2',
    task = 'text=generation',
    provider='auto',
    huggingfacehub_api_token=token
)

llm2 = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Llama-3.1-8B-Instruct',
    task = 'text=generation',
    provider='auto',
    huggingfacehub_api_token=token
)

model = ChatHuggingFace(llm=llm1)

model2 = ChatHuggingFace(llm=llm2)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='give a short notes from the following {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='generate 5 short questions and answers from the following text \n {text}',
    input_variables=['text']
) 

prompt3 = PromptTemplate(
    template='merge the provided notes and quiz in single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['text']
) 

parallel_chain = RunnableParallel({
    'notes': prompt1 | model | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model | parser

chain = parallel_chain | merge_chain

text = """Tesla is a leading electric vehicle manufacturer known for its innovative models like the Model S, Model 3, Model X, and Model Y, offering advanced technology and impressive performance.
Tesla Models
Model S: A luxury all-electric sedan known for its high performance, long range, and advanced technology features. It is designed for speed and efficiency, making it one of the safest and quickest electric cars on the road. 
1
Model 3: A more affordable sedan that has gained popularity for its balance of performance, range, and price. It is designed to be accessible to a broader audience while maintaining Tesla's high standards of quality and technology.
Model X: An all-electric SUV that features distinctive falcon-wing doors, spacious interior, and advanced safety features. It offers a combination of utility and performance, making it suitable for families and adventure seekers alike.
Model Y: A compact SUV that shares many components with the Model 3, offering versatility and space for passengers and cargo. It is currently one of Tesla's most popular models globally.
"""

resutl = chain.invoke({'text':text})

print(resutl)