import joblib
import shap 
import pandas as pd 
import re 
import os 

# Instead of this:
# model = joblib.load("models/news_topic_classifier/news_topic_classifier.pkl")

# Use this:
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, "models", "news_topic_classifier", "news_topic_classifier.pkl"))



vectoriser = joblib.load(os.path.join(BASE_DIR, "models/news_topic_classifier/tfidf_vectorizer.pkl")) 


topics = joblib.load(os.path.join(BASE_DIR, 'models/news_topic_classifier/mlb.pkl'))
topic_hashmap = topics.classes_
model = model.best_estimator_
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
        prediction_df = pd.DataFrame(prediction, columns=topic_hashmap)
        prediction_filtered = prediction_df.loc[:, (prediction_df == 1).all()]
        
        return prediction_filtered
    
        
        
    def predict_proba(self): 
        # this section will generate the prediction probabilities 
        predict_proba = self.model.predict_proba(self.text_tfidf) 
        predict_proba_df = pd.DataFrame(predict_proba, columns=topic_hashmap) 
        prediction_filtered = predict_proba_df.loc[:, (predict_proba_df >= 0.5).all()]
        return prediction_filtered
    
    
    def explain(self): 
        feature_names=vectoriser.get_feature_names_out()
        input_df = pd.DataFrame(self.text_tfidf.toarray(), columns=feature_names)
        explainer = shap.Explainer(self.model) 
        shap_values=explainer(input_df)
        prediction = self.model.predict(self.text_tfidf)[0]


        shap_plot = shap.plots.waterfall(shap_values[:,:,prediction][0])

        # the shap explainer should return the class  
    
     
