from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
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

class review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, 'a brief summary of the review']
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]],'lsit of pros mentioned in the review']

structured_model = model.with_structured_output(review)


result = structured_model.invoke('''The Best Compact Smartphone Of 2025. Performance Top Notch 💪. Battery Life - 85 % Full charge Gives Me Full one day( 24 hrs.) Backup and Still 20-25 % Battery remaining after this and This is Moderate Uses for the phone. Full 100 % Charge Gives One and Half Day Battery backup And Still Battery Remaining At 25 % For Moderate uses of phone. Sometimes It gives Less Battery backup when I used Camera and GPS for Whole day but Other Scenario Battery Life is Awsm'''
)

print(result)