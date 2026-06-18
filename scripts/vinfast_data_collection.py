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
df = pd.read_csv(f"{BASE_DIR}/data/vinfast_data_cleaned.csv") 
df['date'] = pd.to_datetime(df['date']) 

# checking the latest date on df 
day = datetime.datetime.now().day 
month = datetime.datetime.now().month 
year = datetime.datetime.now().year 
current_date = pd.to_datetime(f"{year}-{month}-{day}")
update_date = pd.Timedelta(365, "D") + max(df['date']) 

# deriving the dates and formatting them
from_date = max(df['date'])
from_date = str(from_date).split(" ")[0]
to_date = pd.to_datetime(f'{year}-{month}-{day}')
to_date = str(to_date).split(" ")[0]
# if the update_date is within one week of the current date, I will reload the dataset and append it to the dataframe, saving it 
# from_date and to_date are optional parameters
def collect_historical_data(request:str): 

    try:

        eod_request = requests.get(request)

        print (f"historical vinfast status code: {eod_request.status_code}")

        eod_data = json_normalize(eod_request.json())
        print (f"eod data preview: {eod_data.head()}")
        eod_data['date'] = pd.to_datetime(eod_data['date']) 
        print (min(eod_data['date'])) 
        print (max(eod_data['date'])) 


    except Exception as eod_e: 
        print (f"There was an error: {eod_e}")
    finally: 
        print ("---------------")

    return eod_data 

if pd.to_datetime(f"{year}-{month}-{day}") <= (update_date + pd.Timedelta(7, "D")):
    # updating the date 
    eod_data = collect_historical_data(request=f"https://eodhd.com/api/eod/VFS?api_token={eod_key}&fmt=json")

    # appending to the dataset 
    eod_data = pd.concat([df, eod_data],axis=0)
    eod_data['date'] = pd.to_datetime(eod_data['date']) 
    eod_data.drop_duplicates(inplace=True, subset=['date']) 
    eod_data.to_csv(f"{BASE_DIR}/data/vinfast_data_cleaned.csv") 

else: 
    # In this case, the new dataset does NOT need to be saved. As such, the main streamlit script will automatically append the new data 
    # This should limit the collected data to only the relevant results
    eod_data = collect_historical_data(request=f"https://eodhd.com/api/eod/VFS?api_token={eod_key}&fmt=json&from={from_date}&to={to_date}") 

