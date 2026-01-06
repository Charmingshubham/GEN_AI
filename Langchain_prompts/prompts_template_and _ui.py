from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate, load_prompt
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

token = os.getenv('HUGGING_FACE_API_KEY')

llm = HuggingFaceEndpoint(
    repo_id = 'moonshotai/Kimi-K2-Instruct-0905',
    task = 'text-generation',
    huggingfacehub_api_token=token,
    provider = 'auto'
)

model = ChatHuggingFace(llm=llm)    

st.header('Reaseach tool')

paper_input = st.selectbox(
    'Select your paper',
    ("Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis")
)

style_input = st.selectbox(
    'Select your style',
    ("Formal", "Informal", "Technical", "Simplified")
)

length_input = st.selectbox(
    'Select summary length',
    ("Short", "Medium", "Long")
)

template = load_prompt('template.json')


if st.button('generate'):
    chain  = template | model
    result = chain.invoke({
        'paper_input' : paper_input,
        'style_input' : style_input,
        'length_input' : length_input})
    st.write(result.content)
