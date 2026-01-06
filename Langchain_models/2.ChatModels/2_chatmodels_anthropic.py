from langchain_anthropic import ChatAnthropicAI
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatAnthropicAI(model = 'claude-2', temperature=0.3)

result = chat_model.invoke('What is the capital of Italy?')

print(result.content)