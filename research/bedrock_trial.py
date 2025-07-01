from langchain_community.chat_models import BedrockChat 
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate 
from langchain_core.messages import HumanMessage 
import boto3
import streamlit as st

# Bedrock Client
bedrock_client = boto3.client(
    service_name = "bedrock-runtime",
    region_name = "ap-south-1"
)

# Using Claude 3 Sonnet
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

# Initialize BedrockChat for Claude 3 models
llm = BedrockChat( 
    model_id = model_id,
    client = bedrock_client,
    model_kwargs = {
        "temperature": 0.7,
        "max_tokens": 200 
    }
)

def my_chatbot(language, user_text):
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert assistant. Answer the question in {language} using only 1-2 sentences. Keep it simple. Just return the answer text."
            ),
            ("human", "Question: {user_text}")
        ]
    )

    # For simple Q&A, LLMChain can still work with ChatPromptTemplate and BedrockChat
    bedrock_chain = LLMChain(llm = llm, prompt = chat_prompt)

    response = bedrock_chain.invoke({
        'language': language,
        'user_text': user_text
    })

    return response['text'].strip()


# Streamlit app
st.title("Bedrock Demo Test")

language = st.sidebar.selectbox(
    "Language",
    ['english', 'spanish', 'german']
)

if language:
    user_text = st.sidebar.text_area(
        label = "What is your question?",
        max_chars = 150
    )

    if st.sidebar.button("Ask"): 
        if user_text:
            response = my_chatbot(language.lower(), user_text)
            st.write(response)
        else:
            st.warning("Please enter your question!")

