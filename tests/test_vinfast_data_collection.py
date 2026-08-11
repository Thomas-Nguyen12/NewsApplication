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
from vinfast_data_collection import vinfast_request, eod_request 


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





#  news collection tests 
def test_eod_status_code(): 
    assert eod_request.status_code == 200, "There is an error with the API. The status code is not 200"

def test_news_status_code(): 
    assert vinfast_request.status_code == 200, "The status code should be 200. There is an error with the request"



