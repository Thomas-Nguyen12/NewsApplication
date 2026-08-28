# this page will be for the news page


import joblib 
import pandas as pd
import numpy as np
import re
import shap
import os
import spacy
import time 
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.agents import create_agent 
from langchain.tools import tool 

import streamlit as st

from pathlib import Path 
import plotly.express as px
import sys 
BASE_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = (BASE_DIR / "scripts").resolve()

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# including api key
from preprocessing import lemmatize 
import preprocessing 
from sentiment_analyser import analyser


# agentic ai script



with st.sidebar:
    st.header("Navigation Bar")
# adding a title
st.title("Recent News Articles") 



@st.cache_data 
def sentiment_analysis(text) -> int: 
    text_to_analyse = analyser(text) 
    prediction = np.max(text_to_analyse.predict())
    return prediction 

@st.cache_data
def assign_sentiment_colours(value: int) -> str:
    if value == 0:
        return "🔴"
    elif value == 1:
        return "🟡"
    elif value == 2:
        return "🟢"

@st.cache_data
def assign_sentiment_proba(text:str) -> str:
    text_to_analyse = analyser(text)
    prediction_proba = text_to_analyse.predict_proba()
    print (prediction_proba)
    return f"{round(np.multiply(np.max(prediction_proba), 100))}" 


# api keys are loaded within the scripts.vinfast_data_collection script
# make sure that the paths are correct 


@st.cache_data
def load_news(): 
    from scripts.vinfast_data_collection import load_vinfast_news
    vinfast_news = load_vinfast_news() 
    print ("Analysing the sentiment of the news...") 
    print (vinfast_news.info())
    vinfast_news['title_description_content'] = vinfast_news['title'] + vinfast_news['description'] + vinfast_news['content']
    vinfast_news['title_description_content'] = vinfast_news['title_description_content'].astype(str)
    try: 
        # analysing the content column 
    
        print ("Analysing sentiment...")
        

        # There is no attribute (text)
        #vinfast_news['content_sentiment'] = vinfast_news['title_description_content'].apply(sentiment_analysis)
        # using list comprehension instead of apply
        vinfast_news['content_sentiment'] = [sentiment_analysis(value) for value in vinfast_news['title_description_content']]
        print ("Assigning colours...")
        #vinfast_news['assessment'] = vinfast_news['content_sentiment'].apply(assign_sentiment_colours)
        vinfast_news['sentiment'] = [assign_sentiment_colours(value) for value in vinfast_news['content_sentiment']]

        # There seems to be an issue here

        #vinfast_news['confidence (%)'] = vinfast_news['title_description_content'].apply(assign_sentiment_proba)
        vinfast_news['confidence (%)'] = [assign_sentiment_proba(value) for value in vinfast_news['title_description_content']]
        print (vinfast_news.head()) 
    except Exception as news_e: 
        print (f"There was an exception: {news_e}") 
    finally: 
        print ("Analysis complete!") 

    return vinfast_news 



st.set_page_config(
    page_title="VFS · News Dashboard",
    page_icon="📰",
    layout="wide",
)

# Custom styling 
st.markdown("""
<style>
    /* Dark Vietnamese-flag-inspired palette */
    :root {
        --vf-red:    #D0222A;
        --vf-gold:   #F5C842;
        --vf-dark:   #0F1117;
        --vf-card:   #1A1D27;
        --vf-muted:  #8892A4;
    }

    .stApp { background-color: #0F1117; }

    /* Metric cards */
    .metric-card {
        background: #1A1D27;
        border: 1px solid #2A2D3A;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 4px;
    }
    .metric-label {
        color: #8892A4;
        font-size: 12px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #F0F2F6;
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .metric-delta-pos { color: #3DD68C; font-size: 14px; }
    .metric-delta-neg { color: #D0222A; font-size: 14px; }
</style>
""", unsafe_allow_html=True)


# Loading the news reports
vinfast_news = load_news() 
vinfast_news.columns = vinfast_news.columns.str.strip()
print (f"Column types: {vinfast_news.info()}")

# creating a plot to show the frequency of positive, negative, and bad news 
# I will need to group by the values 
print (vinfast_news)
print ("VINFAST NEWS COLUMNS")
print (vinfast_news.columns)
vinfast_news['publishedAt'] = pd.to_datetime(vinfast_news['publishedAt'])
print ("--------------------------------- VINFAST NEWS")




print (vinfast_news)
grouped_news_sentiment = vinfast_news 
grouped_news_sentiment['publishedAt'] = vinfast_news['publishedAt'].dt.date


grouped_news_sentiment = pd.DataFrame(grouped_news_sentiment.groupby(['publishedAt', 'sentiment'])['sentiment'].count())
grouped_news_sentiment = grouped_news_sentiment.rename({"sentiment": "sentiment_count"},axis=1)

grouped_news_sentiment = grouped_news_sentiment.reset_index()




# I need to manually set the colours
fig = px.bar(grouped_news_sentiment, x='publishedAt', y='sentiment_count', color='sentiment',color_discrete_map={"🟢": "green",
    "🟡": "orange", 
    "🔴": "red"}, 
    barmode='group', title="Recent News Sentiment Frequency",
    labels={'sentiment_count': 'Number of News Articles'})
st.plotly_chart(fig)
# showing the summary of the vinast news reports using agentic ai 


# opening the summary file 
st.header("News Summary using Agentic AI")
st.info("""I am using the free version of news api. As such, there are some issues with creating requests remotely. 
As such, the agentic summary is limited on this website (although it works perfectly fine locally). However, I am developing a workaround""", icon="ℹ️")
st.info("Press the Refresh button to refresh the news summary..", icon="ℹ️")


# BASE_DIR is in the pages/ directory
BASE_DIR2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


vinfast_news_to_display = vinfast_news[['title', 'url', 'sentiment', 'confidence (%)']]

vinfast_news_to_display.index = pd.to_datetime(vinfast_news['publishedAt'])
vinfast_news_to_display = vinfast_news_to_display.sort_index(ascending=False) 

# displaying the table in tabular format

@tool('summarise_vinfast_news_reports', description='summarise The News Reports', return_direct=False)
def summarise_vinfast_news_reports():
    return vinfast_news[['title', 'description', 'content']]


# loading mistral api key
@st.cache_data 
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
    


summariser = build_agent() 







# -------------------
# I need to find a way to limit the number of refreshes
refresh_button=st.button("Refresh news summary")

if refresh_button:
    st.markdown(f"Last Updated: {time.asctime(time.localtime(time.time()) )}")
    news_summary = summariser
    st.markdown(news_summary)
else:
    news_summary = summariser
    st.markdown(news_summary)
# -------------------


st.divider()






with st.expander("View Raw Data"):

    st.dataframe(vinfast_news_to_display) 

