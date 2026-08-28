# this script will use agentic ai to create blog posts of updates and news reports
# Within github pages? 

# There are already some libraries and tools that will summarise text articles
# If i run this script in github actions,



import requests 
from langchain.agents import create_agent 
from langchain.tools import tool 
import streamlit as st 
import pandas as pd


import os
from langchain_groq import ChatGroq
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from vinfast_data_collection import load_vinfast_news
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@tool('summarise_vinfast_news_reports', description='summarise The News Reports', return_direct=False)
def summarise_vinfast_news_reports():
    vinfast_news = load_vinfast_news() 
    vinfast_news = vinfast_news[['title', 'description', 'content']]
    return vinfast_news.to_markdown(index=False)


@tool("summarise_project", description='Summarise the entire project into a report for someone to read, highlighting key skills')
def summarise_project(): 
    # listing all the files and contents (except the hidden files) 
    pass 




# loading mistral api key
def build_agent() -> str:

    GROQ_API_KEY = st.secrets['GROQ_API_KEY']

    # loading prebuild workflows 
    print ("importing vinfast news collection module...")

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="openai/gpt-oss-120b",
        temperature=0,
    )
    print ("Creating the tool...")




    print ("Creating the agent...")
    news_agent = create_agent(
        model=llm,
        tools=[summarise_vinfast_news_reports],
        system_prompt='You are a helpful news assistant who is always friendly.'
    )
    print ("Invoking the response...")
        # testing the code works 
    response = news_agent.invoke({
        'messages': [
            {'role': 'user', 'content': 'summarise, the news reports about vinfast'}
        ]
    })
    print (f"Response type: {type(response)}")
    print ("Showing the response...")
    print (response['messages'][-1]) 


    print ("---------------- DEBUG\n\n\n")
    response_content = dict(response['messages'][-1])
    print (f"response content: {response_content}")
    print (f"response content content: {response_content['content']}")
    print ("------------------\n\n\n")
    print (f"response content content: {type(response_content['content'])}")
    print ("Final output")
    return response_content['content']
    




















# --------------------------------------
@tool("summarise_stock_prices", description='summarise the stock prices changes of the last week')
def summarise_stock_prices():
    # importing the eod data. This data will contain recent stock price data for the past year 
    from vinfast_data_collection import load_eod_data 
    eod_data = load_eod_data()
    return eod_data 


def summarise_recent_stock_prices():
    GROQ_API_KEY = st.secrets['GROQ_API_KEY']

    # loading prebuild workflows 
    print ("This is the summarise_recent_stock_price module")

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model= "openai/gpt-oss-120b",
        temperature=0,
    )
    print ("Creating the tool...")



    print ("Creating the agent...")
    news_agent = create_agent(
        model=llm,
        tools=[summarise_stock_prices],
        system_prompt='You are a helpful stock price assistant who is always friendly and likes to summarise recent stock price changes.'
    )
    print ("Invoking the response...")
        # testing the code works 
    response = news_agent.invoke({
        'messages': [
            {'role': 'user', 'content': 'summarise the stock price changes for the past week'}
        ]
    })
    print ("Showing the response... [response]")
    print (response) 
    print ("-------------------------\n\n\n")
    print ("Showing the response [response['messages']]")
    print (response['messages']) 
    print ("---------------------------\n\n\n\n")
    print ("Showing the response... [response['messages'][-1]]")
    print (response['messages'][-1])
    print ('-----------------------\n\n\n\n')

    return response['messages'][-1].content







    
if __name__ == "__main__": 
    
    print (build_agent())




