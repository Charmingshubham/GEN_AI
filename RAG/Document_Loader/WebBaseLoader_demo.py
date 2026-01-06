from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.document_loaders import TextLoader, WebBaseLoader
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
    template='give me answer of the questions {question} from the following topic {topic}',
    input_variables=['question','topic']
)

url = 'https://www.flipkart.com/samsung-essential-series-s3-55-88-cm-22-inch-full-hd-led-backlit-ips-panel-d-sub-hdmi-flat-monitor-ls22d300gawxxl/p/itm909c8202e1864?pid=MONH7GHGSGF3AGM9&lid=LSTMONH7GHGSGF3AGM9N3DVIC&marketplace=FLIPKART&store=6bo%2Fg0i%2F9no&srno=b_1_1&otracker=browse&otracker1=hp_rich_navigation_PINNED_neo%2Fmerchandising_NA_NAV_EXPANDABLE_navigationCard_cc_3_L2_view-all&fm=organic&iid=en_t6uNkSc3wSEZE7x2LpqqztfmL1efSnKIiMg-boQy7L4CGF63WKXaNuiGJrVv2BZH2yIzfxSZJRf11sSBknI91w%3D%3D&ppt=hp&ppn=homepage&ssid=4r7cre5r5c0000001763960556743'

loader = WebBaseLoader(url)

docs = loader.load()

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'question':'what is the product we are talking about','topic':docs[0].page_content})

print(result)