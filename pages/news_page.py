# this page will be for the news page


import joblib 
import pandas as pd
import numpy as np
import re
import shap
import os
import spacy


import streamlit as st

from pathlib import Path 
import plotly.express as px
import sys 
BASE_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = (BASE_DIR / "scripts").resolve()

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from preprocessing import lemmatize 
import preprocessing 
from sentiment_analyser import analyser


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
    from scripts.vinfast_data_collection import vinfast_news 
    print ("Analysing the sentiment of the news...") 
    try: 
        # analysing the content column 
    
        print ("Analysing sentiment...")

        # There is no attribute (text)
        vinfast_news['content_sentiment'] = vinfast_news['content'].apply(sentiment_analysis)
        print ("Assigning colours...")
        vinfast_news['sentiment'] = vinfast_news['content_sentiment'].apply(assign_sentiment_colours)
        # There seems to be an issue here

        vinfast_news['confidence (%)'] = vinfast_news['content'].apply(assign_sentiment_proba)

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


# creating a plot to show the frequency of positive, negative, and bad news 
# I will need to group by the values 
vinfast_news['publishedAt'] = pd.to_datetime(vinfast_news['publishedAt'])
grouped_news_sentiment = vinfast_news 
grouped_news_sentiment['publishedAt'] = vinfast_news['publishedAt'].dt.date


grouped_news_sentiment = pd.DataFrame(grouped_news_sentiment.groupby(['publishedAt', 'sentiment'])['sentiment'].count())
grouped_news_sentiment = grouped_news_sentiment.rename({"sentiment": "sentiment_count"},axis=1)

grouped_news_sentiment = grouped_news_sentiment.reset_index()




# I need to manually set the colours
fig = px.bar(grouped_news_sentiment, x='publishedAt', y='sentiment_count', color='sentiment', barmode='group',
    color_discrete_sequence=['orange', 'red', 'green'], title="Recent News Sentiment Frequency",
    labels={'sentiment_count': 'Number of News Articles'})
st.plotly_chart(fig)


vinfast_news_to_display = vinfast_news[['title', 'url', 'sentiment', 'confidence (%)']]

vinfast_news_to_display.index = pd.to_datetime(vinfast_news['publishedAt'])
vinfast_news_to_display = vinfast_news_to_display.sort_index(ascending=False) 
# displaying the table in tabular format





st.dataframe(vinfast_news_to_display) 
