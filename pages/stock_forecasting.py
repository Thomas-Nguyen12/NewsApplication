import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt 
from pathlib import Path
import os 
import numpy as np
import joblib 
import plotly.graph_objects as go 
import plotly.express as px
import sys 
import pandas_market_calendars as mcal
from datetime import time, datetime, timedelta
# Create a calendar (vinfast is on nasdaq)
nasdaq = mcal.get_calendar('NASDAQ')

# Show available calendars
print (mcal.get_calendar_names())

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print (f"BASE DIR: {BASE_DIR}")
SCRIPTS_DIR = f"{BASE_DIR}/scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

st.title("Vinfast Stock Forecasting")
st.subheader("Please note: This section is still being refined...")
# I can include a streamlit slider 
with st.sidebar: 
    st.header("Navigation Bar")


# I will use the interactive plot library plotly
# ------------------- creating the set of predictions 

from vinfast_multiple_predictors import forecaster



# Collecting live data 
@st.cache_data
def load_news_eod(): 
    # depending on the results from vinfast_data_collection.py, the eod_data will either 
    # be an updated dataset or not 
    #
    from scripts.vinfast_data_collection import load_eod_data
    


    return load_eod_data()
eod_data = load_news_eod() 
df = eod_data
# joining the data 



max_date = max(df['date']) 
min_date = min(df['date']) 


# I need to account for holidays and sotck 

market_dates = nasdaq.schedule(start_date=max_date, end_date=(max_date + pd.Timedelta(value=30, unit="D")))

market_date_range = mcal.date_range(market_dates, frequency="1D")
print (f"MARKET DATE RANGE: {market_date_range}")
df = df.sort_index(ascending=True)
# I can now create the slider using the market dates
print (f"MIN MARKET DATE: {min(market_date_range)}")
print (f"MAX MARKET DATE: {max(market_date_range)}")

minimum_market_date = (min(market_date_range))
minimum_market_date = datetime(minimum_market_date.year, minimum_market_date.month, minimum_market_date.day)



maximum_market_date = (max(market_date_range)) 
maximum_market_date = datetime(maximum_market_date.year, maximum_market_date.month, maximum_market_date.day)



# maximum and minimum market date range are pandas date times



starting_date = (minimum_market_date + timedelta(days=2))

date_range = st.slider(
    'Date Range',
    min_value=(minimum_market_date),
    max_value=(maximum_market_date),
    value=(starting_date),
    format="MMM DD, YYYY"


)
st.write(f"date range: {date_range}")
st.write (f"n periods: {((date_range - minimum_market_date).days)}")
# n_periods can be calculated by finding the difference between 
# I can get the stock data by accessing the df

open_ = df.tail(1)['open'].values[0]
close_ = df.tail(1)['adjusted_close'].values[0]
high_ = df.tail(1)['high'].values[0]
low_ = df.tail(1)['low'].values[0]


test = forecaster(
    input_date           = pd.to_datetime(minimum_market_date),
    n_periods             = (date_range - minimum_market_date).days,
    seed_open             = open_,
    seed_high             = high_,
    seed_low              = low_, 
    seed_adjusted_close   = close_,   
)
# running the forecaster
forecast = test.forecast()
st.divider()
st.header("Forecast")
st.dataframe(forecast)

# to plot all data together, I may need to reshape the data 
forecast_melted = pd.melt(forecast, id_vars=['date'], value_vars=['predicted_open', 'predicted_adjusted_close', 'predicted_high', 'predicted_low'])
forecast_melted.columns = ['date', 'category', 'value']



# plotting 


fig = px.line(forecast_melted, x="date", y='value', color='category')
st.plotly_chart(fig, config = {'scrollZoom': False})
