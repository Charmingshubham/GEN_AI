from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI()

class review(BaseModel):
    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description='a brief summary of the review')
    sentiment: Literal['pos','neg'] = Field(description="Return sentiment of the review either negative, positive or neutral")
    pros : Optional[list[str]] = Field(description='lsit of pros mentioned in the review')

structured_model = model.with_structured_output(review)


result = structured_model.invoke('''The Best Compact Smartphone Of 2025. Performance Top Notch 💪. Battery Life - 85 % Full charge Gives Me Full one day( 24 hrs.) Backup and Still 20-25 % Battery remaining after this and This is Moderate Uses for the phone. Full 100 % Charge Gives One and Half Day Battery backup And Still Battery Remaining At 25 % For Moderate uses of phone. Sometimes It gives Less Battery backup when I used Camera and GPS for Whole day but Other Scenario Battery Life is Awsm'''
)

print(result)

#cannot use huggingface model for structured output with pydantic, but it is possible with openai models
#because the openai models supports with_structured_output function which makes the output to be structered
#and the open source model do nnot have this feature because they small maodels and the openai is fine tunned and larger model

