import pandas as pd 
import os 
import sys 
import datetime 
import re

""" 
Within this script, I will run the scraper and clean the collected data, naming it as the timeframe it captures.

"""

# ensuring the correct directory to the main "news_project/" folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

current_working_directory = os.getwd() 

if current_working_directory != BASE_DIR:
    print ("Incorrect Working Directory...")
    print ("Adjusting...") 
    os.chdir(BASE_DIR) 
else:
    print ("Correct Working Directory...") 







current_date = datetime.datetime.now()
current_year = current_date.year
current_month = current_date.month
current_day = current_date.day



# This is the path to the scraper


sys.path.append("news_scraper/news_scraper/scraper/")


print ("Running the scraper...")
try:

    os.system(f"scrapy crawl scraper -o financial_data/news/news_{current_day}_{current_month}_{current_year}.csv")
except as e:
    print (f"There was an error: {e}")



# accessing the dataset to clean it

df = pd.read_csv(f"data/news_{current_day}_{current_month}_{current_year}.csv")


# dropping unecessary columns 
df.drop(['headings'], axis=1, inplace=True) 

# formatting the date column
df['date'] = pd.to_datetime(df['date']) 

# cleaning hte text in the "text" column


## removing edit,history,watch
df['text'] = df['text'].str.replace("^edit,history,watch", "") 

## removing the \n
df['text'] = df['text'].str.replace("\n", "") 


# sorting the table by date
df = df.sort_values(['date'], ascending=True) 

# removing incorrect dates...
if df.tail(1).date != pd.to_datetime(f"{current_year}-{current_month}-{current_date}"):
    print ("The latest date does not match. Removing...") 
    df[df.date != pd.to_datetime(f"{current_year}-{current_month}-{current_day}")]
else:
    print ("The dates are okay...") 





# cleaning the "topic" column 




## 2. checking for spelling differences. This may need an approximation value to the closest matching topic
# This may involve checking which ones i removed...


"""
The correct ones:
armed conflicts and attacks
law and crime and politics
disasters and accidents
politics and elections and economics
international relations
health and environment
business and economics
sports
science and technology
arts and culture

"""
replacements = {
    "Armed conflicts and attacks": "armed conflicts and attacks",
    "Disasters and accidents": "disasters and accidents",
    "Law and crime": "law and crime and politics",
    "Politics and elections": "politics and elections and economics",
    "International relations": "international relations",
    "Business and economy": "business and economics",
    "Sports": "sports",
    "Science and technology": "science and technology",
    "Health and environment": "health and environment",
    "Arts and culture": "arts and culture",
    # Typos / variants
    "Science and Technology": "science and technology",
    "Disaster and accidents": "disasters and accidents",
    "Arts and Culture": "arts and culture",
    "Business and econony": "business and economics",
    "Attacks and armed conflicts": "armed conflicts and attacks",
}

df['topic'] = df['topic'].replace(replacements, regex=False)



df['topic'] = df['topic'].str.lower()



df.to_csv(f"financial_data/news/news_{current_day}_{current_month}_{current_year}")



