import pandas as pd 
import datetime 
import numpy as np 
import os 
from pandas import json_normalize
import requests 
import streamlit as st 
from preprocessing import lemmatize



# loading the sentiment analysis script 
# I need to make sure that the script loads the correct directory
# I cannot use the sys module 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# testing


# loading API keys

df = pd.read_csv(f"{BASE_DIR}/data/vinfast_data_cleaned.csv")

eod_key = st.secrets["fin_historical_data"]
news_api_key = st.secrets['news_api_key']
# collecting news data 
#

try:
    vinfast_request = requests.get(f"https://newsapi.org/v2/everything?q=vinfast&apiKey={news_api_key}&language=en")
    print (f"News status code: {vinfast_request.status_code}")
    vinfast_news = json_normalize(vinfast_request.json()['articles'])

    print ("Extracting article text...") 

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
    df['date'] = pd.to_datetime(df['date'])
    print ("checking if the historical eod needs updating...")
    # ------------------- UPDATING THE DATASET IF NEEDED
    if (max(df['date']) + pd.Timedelta(350, unit="D")) > max(eod_data['date']):
        print ("Updating the saved dataset using the eod data") 
        # update df and save it
        df = pd.concat([df, eod_data],axis=0) 
        df['date'] = pd.to_datetime(df['date']) 
        df.to_csv("data/vinfast_data_cleaned.csv", index=False) 
        
    else:
        print ("No need to update the eod data")
        eod_data = pd.concat([df, eod_data], axis=0)
        eod_data['date'] = pd.to_datetime(eod_data['date'])

        # dropping the duplicates 
        eod_data = eod_data.drop_duplicates(subset=['date']) 


except Exception as eod_e: 
    print (f"There was an error: {eod_e}")
finally: 
    print ("---------------") 

