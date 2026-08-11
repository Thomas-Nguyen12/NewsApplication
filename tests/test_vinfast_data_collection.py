# this script will test the output of the scripts/vinfast_data_collection.py script


import sys 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import scipy.stats as stats 
import pytest 

# adding the scripts folder
#
sys.path.append("../scripts/") 
# testing that the script import works 

from vinfast_data_collection import vinfast_news, eod_data, vinfast_request 


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


def test_news_dataframe(): 
    columns = ['author', 'title', 'description', 'url', 'urlToImage', 'publishedAt',
               'content', 'source.id', 'source.name'],
    
    for column in columns: 
        assert column in columns, "There is an improperly formatted column within the vinfast news"
def test_news_status_code(): 
    assert vinfast_request.status_code == 200, "The status code should be 200. There is an error with the request"



