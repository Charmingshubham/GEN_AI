import os
from dotenv import load_dotenv

load_dotenv()

print("HF TOKEN:", os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"))
