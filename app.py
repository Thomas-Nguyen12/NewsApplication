import dash 
import dash_html_components as html 
import dash_core_components as dcc 
from dash import Dash, html, dcc, callback, Output, Input
import joblib
# This module is for graphing 
import plotly.express as px
import pandas as pd
import plotly.express as px
import numpy as np
<<<<<<< HEAD

# this is the ai detection code
from ai_detection import ai_detector
from sklearn.ensemble import RandomForestClassifier 

# this is the news classifier
from classify_news import classifier


=======
import sklearn.metrics.pairwise.cosine_similarity
>>>>>>> cbe3dd226a5256d3ec9498dd10589d1464e408c0
# Requires Dash 2.17.0 or later
"""
app.layout() --> describes the layout and appearance of the app

I will use the 

"""

<<<<<<< HEAD

extracted_topics = pd.read_csv("data/news_data_ready_to_plot.csv")
unique_topics=extracted_topics.topic_split.unique()
=======
news_model = joblib.load("news_model.pkl")
>>>>>>> cbe3dd226a5256d3ec9498dd10589d1464e408c0




<<<<<<< HEAD
=======


# clean_df will contain the date and text 
clean_df = pd.read_csv("clean_df.csv") 


# extracted topics will contain the numerical count information for topics and dates 
extracted_topics = pd.read_csv("extracted_topics.csv") 

extracted_topics.drop(['Unnamed: 0'], axis=1, inplace=True)
unique_topics = extracted_topics['topic'].unique() 
tfidf_vectoriser = joblib.load('tfidf_vectoriser.pkl')


>>>>>>> cbe3dd226a5256d3ec9498dd10589d1464e408c0
app = Dash()

# visualising the data (news trends over time, but the user chooses the topic)

# app.layout contains an array of divs that contain the interactive elements




app.layout = html.Div([
    html.H1(children='News Trends over time'),
<<<<<<< HEAD

    html.Div(children=[
        html.Label('Choose a news category'),
        dcc.Dropdown([*unique_topics], 'Law and crime', multi=True, id='topic-dropdown'),
        
        html.Br(),

=======
    
    
    html.Div(children=[
        html.Label('Choose a news category'),
        dcc.Dropdown([*unique_topics], 'Law and crime', multi=False, id='topic-dropdown'),
        
        html.Br(),
        
        
        
>>>>>>> cbe3dd226a5256d3ec9498dd10589d1464e408c0
        dcc.Graph(
            id='news-graph',
            figure={}
            
        ),
<<<<<<< HEAD
=======

         
        
>>>>>>> cbe3dd226a5256d3ec9498dd10589d1464e408c0
    ]),
    
    html.H1(children='News Classifier'),
    html.Div(children=[html.Label("This model will classifier will group news reports between 14 different topics")]),
    
    html.Div(children=[
        dcc.Textarea(
        id='text-box',
        value='Enter your news here',
        style={'width': '100%', 'height': 300}),
        
<<<<<<< HEAD
    ]),
    # this is an output
    html.Div(id='classifier-output'),  
    
    html.Div(id='ai-detection-output')
])
# -------------------------- Creating a figure
=======
        
    ]),
    
    html.Div(id='classifier-output')    
])


# ---------------------
>>>>>>> cbe3dd226a5256d3ec9498dd10589d1464e408c0
@callback(
    Output('news-graph', 'figure'),
    Input('topic-dropdown', 'value') 
)
<<<<<<< HEAD

def update_figure_topic(input_topic): 
    # Filtering the data to the choice
    """
=======
def update_figure_topic(input_topic): 
>>>>>>> cbe3dd226a5256d3ec9498dd10589d1464e408c0
    filtered_topic_df = extracted_topics[extracted_topics.topic == input_topic] 
    fig = px.line(filtered_topic_df, x='year_month', y='count', color='topic') 
    fig.update_layout(transition_duration=500) 
    return fig
<<<<<<< HEAD
    """
    
    filtered_topic_df = extracted_topics[extracted_topics.topic_split.isin(input_topic)]
    fig = px.line(filtered_topic_df, x='date_cleaned', y='count', color='topic_split')
    fig.update_layout(transition_duration=500)
    return fig
    
    
# ---------------------- Implemeneting the model here. This part works fine
=======
    
# ---------------------- Implemeneting the model here
>>>>>>> cbe3dd226a5256d3ec9498dd10589d1464e408c0
@callback(
    Output('classifier-output', 'children'),
    Input('text-box', 'value')
)

def classify_news(input_news):
    
    # vectorising the text
<<<<<<< HEAD
    
    text_to_classify = classifier(text=input_news) 

    prediction = text_to_classify.predict() 
    prediction_probability = text_to_classify.predict_proba() 
    


=======
    input_news = [input_news]
    
    
    vectorised_text = tfidf_vectoriser.transform(input_news) 
    
    
    # passing the vectorised text into the model
    prediction = news_model.predict(vectorised_text).toarray()
    
    
    


    # finding out what the news category belonged to 
    prediction_df = pd.DataFrame(prediction, columns=extracted_topics.topic.unique())
    

>>>>>>> cbe3dd226a5256d3ec9498dd10589d1464e408c0

    # returning the output
    
    # Maybe I can format the results so that only columns with a one in it are returned
<<<<<<< HEAD

    
    return f"The prediction is: {prediction}... with {prediction_probability} confidence (out of 1)"


# This area will be for calculating similarity with known Ai models

# ----------------------- detecting ai generated news
@callback(
    Output('ai-detection-output', 'children'),
    Input("text-box", 'value') 
)
def detect_ai(input_news): 
    # vectorising the text
    # the module is "ai_detection" 
    detect_text = ai_detector(text=input_news) 
    prediction = detect_text.predict() 
    prediction_probability = detect_text.predict_proba() 
    
    return f"{prediction} with confidence: {prediction_probability} (out of 1)"
=======
    selected_cols = prediction_df.columns[prediction_df.iloc[0] == 1].tolist()
    
    
    return f"The prediction is: {selected_cols}..."


# This area will be for calculating similarity with known Ai models
def cosine_similarity(input_text: str) -> str:
    
    """
    (1) Pass the news report into the AI report
    (2) Assess similarity with AI models
    """
     pass

>>>>>>> cbe3dd226a5256d3ec9498dd10589d1464e408c0

if __name__ == "__main__":
    app.run(debug=True, port=1234) 
    
    