import joblib
import shap 
import pandas as pd 
import re 
vectoriser = joblib.load("data/tfidf_vectoriser.pkl") 

topic_hashmap = joblib.load('data/topic_hashmap.pkl')
model = joblib.load("data/news_topic_classifier.pkl") 
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
        prediction_df = pd.DataFrame(prediction.toarray(), columns=topic_hashmap.keys())
        prediction_filtered = prediction_df.loc[:, (prediction_df == 1).all()]
        
        return prediction_filtered
    
        
        
    def predict_proba(self): 
        # this section will generate the prediction probabilities 
        predict_proba = self.model.predict_proba(self.text_tfidf) 
        predict_proba_df = pd.DataFrame(predict_proba.toarray(), columns=topic_hashmap.keys()) 
        prediction_filtered = predict_proba_df.loc[:, (predict_proba_df >= 0.5).all()]
        return prediction_filtered
    
    
    def explain(self): 
        """
        print (f"model: {self.model}")
        explainer = shap.Explainer(self.model.predict_proba, feature_names=self.vectoriser.get_feature_names_out())
        shap_values = explainer(self.text_tfidf.toarray())
        
        # bear in mind, you can specify different classes using 
        bar_plot = shap.plots.bar(shap_values[:,:,0][0])
        return bar_plot
        """
        pass
    
     