import joblib
import numpy 
import shap
import re 
import pandas as pd



model = joblib.load("models/sentiment_analyser/sentiment_analyser.pkl")
vectoriser = joblib.load("models/sentiment_analyser/sentiment_vectoriser.pkl") 


class classifier:


    def __init__(self, text:str):

        self.text = re.sub('[^a-zA-Z.]', text).strip()
        self.text_tfidf=vectoriser.transform([text])

    
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

