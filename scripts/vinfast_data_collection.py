import pandas as pd 
import datetime 
import numpy as np 
import os 
from pandas import json_normalize
import requests 
import streamlit as st 
from scripts.preprocessing import lemmatize

# loading the sentiment analysis script 
# I need to make sure that the script loads the correct directory
# I cannot use the sys module 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# testing

""" 

This script will collect finance data 
and financial news data for vinfast 

This script will be a module that will be imported by the main streamlit app 
NOTE THAT i cannot use dotenv in streamlit 

"""

# loading API keys

eod_key = st.secrets["fin_historical_data"]
news_api_key = st.secrets['news_api_key']
# collecting news data 
#

try:
    vinfast_request = requests.get(f"https://newsapi.org/v2/everything?q=vinfast&apiKey={news_api_key}&language=en")
    print (f"News status code: {vinfast_request.status_code}")
    vinfast_news = json_normalize(vinfast_request.json()['articles'])
    print (f"Vinfast preview: {vinfast_news.head()}")

except Exception as e: 
    print (f"There was an error: {e}")
finally:
    print ("----------------") 


# ---------------------- Loading historical stock data 
#
try:

    eod_request = requests.get(f"https://eodhd.com/api/eod/VFS?api_token={eod_key}&fmt=json")

    print (f"historical vinfast status code: {eod_request.status_code}")

    eod_data = json_normalize(eod_request.json())
    print (f"eod data preview: {eod_data.head()}")
    eod_data['date'] = pd.to_datetime(eod_data['date']) 

except Exception as eod_e: 
    print (f"There was an error: {eod_e}")
finally: 
    print ("---------------") 

