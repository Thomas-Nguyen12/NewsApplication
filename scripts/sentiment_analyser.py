
from preprocessing import lemmatize 
import preprocessing
print (f"preprocessing: {preprocessing}")

import joblib
import pandas as pd
import numpy
import re
import shap
import os

import spacy

# Use this:

nlp = spacy.load("en_core_web_sm")
stop_words = list(nlp.Defaults.stop_words) 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, "models/sentiment_analyser/sentiment_analyser.pkl"))
vectoriser = joblib.load(os.path.join(BASE_DIR, "models/sentiment_analyser/sentiment_vectoriser.pkl"))


                         
class analyser:
    


    def __init__(self, text:str):

        self.text = re.sub('[^a-zA-Z]', ' ', self.text).strip()

        # the vectoriser needs to access the lemmatizer
        self.text_tfidf=vectoriser.transform([self.text])

    
    def predict(self):
        prediction = model.predict(self.text_tfidf)
        return prediction 

    def predict_proba(self):
        prediction_proba = model.predict_proba(self.text_tfidf)
        return prediction_proba

    def explain(self):
       
       feature_names = vectoriser.get_feature_names_out()
       input_df=pd.DataFrame(self.text_tfidf.toarray(), columns=feature_names)
       explainer = shap.Explainer(model)

       shap_values = explainer(input_df) 

       prediction = model.predict(self.text_tfidf)[0]

       shap_plot = shap.plots.waterfall(shap_values[:,:,prediction][0])
       return shap_plot
    

if __name__ == '__main__': 
    print ("This is the main file") 
    print (f"Base dir: {BASE_DIR}")
    print (f"Lemmatizer: {lemmatize}")
