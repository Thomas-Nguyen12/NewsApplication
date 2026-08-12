# this script will test the output of the scripts/vinfast_data_collection.py script


import sys 
import pandas as pd 
import numpy as np 
import pytest 
import streamlit as st 
import os


# adding the scripts folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(f"{BASE_DIR}/scripts/") 
# testing that the script import works 
# make sure the module is found
from vinfast_data_collection import load_vinfast_news, load_eod_data, check_vinfast_status_code, check_eod_status_code 




"""
TESTS: 


# News collection 
1. check the corect columns
2. check the dataframe is not empty 
3. check the status code 
4. check the exceptions 
5. 



# EOD Data 
# 1. check the correct columns 
# 2. check the dataframe is not empty 
# 3. check that status code 
# 4. check the exceptions 
"""



@pytest.fixture 
def show_eod_data(): 
    return load_eod_data() 

@pytest.fixture 
def show_vinfast_news(): 
    return load_vinfast_news()

@pytest.fixture 
def show_vinfast_status_code(): 
    return check_vinfast_status_code()

@pytest.fixture 
def show_eod_status_code(): 
    return check_eod_status_code() 



#  news collection tests 


def test_eod_status_code(show_eod_status_code): 
    assert show_eod_status_code == 200, "There was an error with the EOD data API. The status code should be 200"

def test_vinfast_news_status_code(show_vinfast_status_code): 
    assert show_vinfast_status_code == 200, "There was an error with the Vinfast news API. The status code should be 200"


def test_eod_dataframe(show_eod_data): 
    assert isinstance(show_eod_data, pd.DataFrame), "There is an error with the API. the EOD data should be a dataframe"

def test_vinfast_dataframe(show_vinfast_news): 
    assert isinstance(show_vinfast_news, pd.DataFrame), "There is an error with the API or data cleaning. Vinfast news should be a dataframe"


