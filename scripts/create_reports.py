# this script will use agentic ai to create blog posts of updates and news reports
# Within github pages? 

# There are already some libraries and tools that will summarise text articles
# If i run this script in github actions,



import requests 
from langchain.agents import create_agent 
from langchain.tools import tool 
import streamlit as st 


import os
from langchain_groq import ChatGroq
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from vinfast_data_collection import load_vinfast_news
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@tool('summarise_vinfast_news_reports', description='summarise The News Reports from The Vinfast News API', return_direct=False)
def summarise_vinfast_news_reports():
    vinfast_news = load_vinfast_news() 
    return vinfast_news

# loading api keys 

# loading mistral api key
def build_agent():

    GROQ_API_KEY = st.secrets['GROQ_API_KEY']

    # loading prebuild workflows 
    print ("importing vinfast news collection module...")

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="qwen/qwen3.6-27b",
        temperature=0,
    )
    print ("Creating the tool...")




    print ("Creating the agent...")
    news_agent = create_agent(
        model=llm,
        tools=[summarise_vinfast_news_reports],
        system_prompt='You are a helpful news assistant who is always friendly. You are usually provided with a news report dataframe'
    )
    print ("Invoking the response...")
        # testing the code works 
    response = news_agent.invoke({
        'messages': [
            {'role': 'user', 'content': 'Summarise the news reports about vinfast. The news reports are in a dataframe'}
        ]
    })
    print ("Showing the response...")
    print (response['messages'][-1].content)


    print ("Saving the summary to a file...")

    

    with open(f"{BASE_DIR}/data/vinfast_news_data/summarised_news_articles.md", "w") as f: 
        f.write(response['messages'][-1].content)
        f.close() 
    

    return response['messages'][-1].content






if __name__ == "__main__": 
    build_agent() 

