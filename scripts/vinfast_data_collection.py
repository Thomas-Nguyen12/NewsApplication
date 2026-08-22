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

# testing


# loading API keys


# collecting news data 
#
#
#
news_api_key = st.secrets['news_api_key']
eod_key = st.secrets["fin_historical_data"]
def check_vinfast_status_code(): 

        
        vinfast_request = requests.get(f"https://newsapi.org/v2/everything?q=vinfast&apiKey={news_api_key}&language=en")
        return vinfast_request.status_code 




def load_vinfast_news(): 

    try:


        
        vinfast_request = requests.get(f"https://newsapi.org/v2/everything?q=vinfast&apiKey={news_api_key}&language=en")
        vinfast_news = json_normalize(vinfast_request.json()['articles'])


        return vinfast_news 
    except Exception as e: 
        print (f"There was an error: {e}")
    

# ---------------------- Loading historical stock data 
#
#
#
def check_eod_status_code(): 
        
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        df = pd.read_csv(f"{BASE_DIR}/data/vinfast_data_cleaned.csv")
        eod_request = requests.get(f"https://eodhd.com/api/eod/VFS?api_token={eod_key}&fmt=json")
        return eod_request.status_code 



def load_eod_data(): 

    try:
        
        
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        df = pd.read_csv(f"{BASE_DIR}/data/vinfast_data_cleaned.csv")
        eod_request = requests.get(f"https://eodhd.com/api/eod/VFS?api_token={eod_key}&fmt=json")


        eod_data = json_normalize(eod_request.json())
        eod_data['date'] = pd.to_datetime(eod_data['date']) 
        df['date'] = pd.to_datetime(df['date'])
        # ------------------- UPDATING THE DATASET IF NEEDED
        if (max(df['date']) + pd.Timedelta(350, unit="D")) < max(eod_data['date']):
            # update df and save it
            df = pd.concat([df, eod_data],axis=0) 
            df['date'] = pd.to_datetime(df['date']) 
            df = df.drop_duplicates(subset=['date'])
            df.to_csv(f"{BASE_DIR}/data/vinfast_data_cleaned.csv", index=False) 
            return df  
        else:
            eod_data = pd.concat([df, eod_data], axis=0)
            eod_data['date'] = pd.to_datetime(eod_data['date'])

            # dropping the duplicates 
            eod_data = eod_data.drop_duplicates(subset=['date']) 
            return eod_data


    except Exception as eod_e: 
        print (f"There was an error: {eod_e}")





if __name__ == "__main__":
    print ("Loading EOD data...")
    load_eod_data()
    print ("Loading vinfast news...")
    load_vinfast_news()
