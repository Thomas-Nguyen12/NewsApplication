import joblib
import pandas as pd
import numpy
import re
import shap
import os
import spacy
import __main__  # ← add this

nlp = spacy.load("en_core_web_sm")

# 1. Define lemmatize FIRST
def lemmatize(text):
    doc = nlp(text)
    tokens = [token for token in doc if not token.is_punct]
    lemmas = [token.lemma_ if token.pos_ != 'PRON' else token.orth_ for token in tokens]
    return lemmas

# 2. Patch __main__ so pickle finds it
__main__.lemmatize = lemmatize


# NOW load the models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
model = joblib.load(os.path.join(BASE_DIR, "models/news_topic_classifier/news_topic_classifier.pkl"))
vectoriser = joblib.load(os.path.join(BASE_DIR, "models/news_topic_classifier/tfidf_vectorizer.pkl"))
topics = joblib.load(os.path.join(BASE_DIR, 'models/news_topic_classifier/mlb.pkl'))

class classifier: 
    
    def __init__(self, text:str): 
        self.text=re.sub("[^a-zA-Z0-9]+", " ", text).strip() 
        
        
    
        self.vectoriser = vectoriser
        self.text_tfidf = self.vectoriser.transform([self.text])
        print (f"text tfidf: {self.text_tfidf}")
        self.model = model

        
        
        
    
    def predict(self):
        # this section will be for predicting the labels
        prediction = self.model.predict(self.text_tfidf)
        prediction_df = pd.DataFrame(prediction, columns=topics.classes_)
        prediction_filtered = prediction_df.loc[:, (prediction_df == 1).all()]
        
        return prediction_filtered
    
        
        
    def predict_proba(self): 
        # this section will generate the prediction probabilities 
        predict_proba = self.model.predict_proba(self.text_tfidf) 
        predict_proba_df = pd.DataFrame(predict_proba, columns=topics.classes_) 
        prediction_filtered = predict_proba_df.loc[:, (predict_proba_df >= 0.5).all()]
        return prediction_filtered
    
    
    def explain(self): 
        feature_names=vectoriser.get_feature_names_out()
        input_df = pd.DataFrame(self.text_tfidf.toarray(), columns=feature_names)
        explainer = shap.KernelExplainer(self.model.predict_proba, data=) 
        shap_values=explainer(input_df)
        prediction = self.model.predict(self.text_tfidf)[0]


        shap_plot = shap.plots.waterfall(shap_values[:,:,prediction][0])

        # the shap explainer should return the class  
    
     
