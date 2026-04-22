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

current_working_directory = os.getcwd() 

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


os.chdir("news_scraper/news_scraper/spiders/")


print ("Running the scraper...")
try:

    os.system(f"scrapy crawl scraper -o ../../../financial_data/news/news_{current_day}_{current_month}_{current_year}.csv")
except Exception as e:
    print (f"There was an error: {e}")

os.chdir("../../../")

# accessing the dataset to clean it

df = pd.read_csv(f"financial_data/news/news_{current_day}_{current_month}_{current_year}.csv")


# dropping unecessary columns 
df.dropna(axis=1, how='all', inplace=True)



# cleaning hte text in the "text" column


## removing edit,history,watch
print ("Removing edit history watch")
df['text'] = df['text'].str.replace("^edit,history,watch,", "", regex=True) 

## removing the \n
df['text'] = df['text'].str.replace("\n", "") 


# sorting the table by date
df = df.sort_values(['date'], ascending=True) 




# removing incorrect dates...
print (f"missing... {df.isna().sum()}")
print (f"missing... {df[df.topic.isna()]}")
print ("Removing...")
df.dropna(inplace=True)
print (f"missing... {df.isna().sum()}")

df = df[df['date'] != 'date']
df['date'] = pd.to_datetime(df['date'], format='%B %d, %Y')



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

df['topic'] = df['topic'].replace(replacements, regex=True)



df['topic'] = df['topic'].str.lower()
replacements2 = {
    "disaster and accidents": "disasters and accidents",

    "attacks and armed conflicts": "armed conflicts and attacks",
    "business and econony": "business and economics",
}
df['topic'] = df['topic'].replace(replacements2, regex=True)



print (f"The data spans from... {df.date.min()} to {df.date.max()}")


df.to_csv(f"financial_data/news/news_{current_day}_{current_month}_{current_year}.csv", header=True, index=False)



