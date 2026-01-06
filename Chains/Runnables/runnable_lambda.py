from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv  
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel ,RunnableSequence, RunnableLambda,RunnablePassthrough
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

parser = StrOutputParser

def word_count(text):
    return len(text.split())

prompt = PromptTemplate(
    template='write a joke about {topic}',
    input_variables=['topic']
)

joke_gen_chain = RunnableSequence(prompt,model,parser)

parallelchain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'word_count' : RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_gen_chain,parallelchain)

result = final_chain.invoke({'topic':'AI'})

finalresult = '''{} \n word_count {}'''.format(result['joke'],result['word_count'])

print(finalresult)