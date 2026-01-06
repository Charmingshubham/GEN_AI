from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding_model = OpenAIEmbeddings(model = "text-embedding-ada-002",dmiemnsions=100)

result = embedding_model.embed_query('what is the capital of france?')

print(str(result))