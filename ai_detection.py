import pandas as pd
import joblib
import numpy as np
import shap
import re
vectoriser = joblib.load("arvix_data/ai_vectoriser.pkl")
model = joblib.load("arvix_data/ai_detector.pkl")
class ai_detector:
    def __init__(self, text:str):

        # store input text
        self.text = re.sub("[^a-zA-Z0-9]+", " ", text).strip()
        

        # load TF-IDF vectorizer
        
        
        # removing unecessary special characters
        self.vectoriser = vectoriser
        self.model = model 
        
        

        # transform text
        self.text_list = [self.text]
        self.text_tfidf = self.vectoriser.transform(self.text_list)

        # load model
        



    def predict(self):
        prediction = self.model.predict(self.text_tfidf)
        if prediction == 0: 
            return "This is likely a human generated text"
        
        
        else:
            return "This is likely an AI generated text"


    def predict_proba(self):
        prediction_proba = self.model.predict_proba(self.text_tfidf)
        prediction_probabilities = pd.DataFrame(prediction_proba, columns=['human_generated', 'AI_generated'])
        prediction_probabilities_filtered = prediction_probabilities.loc[:, (prediction_probabilities >= 0.5).all()]
        return prediction_probabilities_filtered.values


    def explain(self):
        # create SHAP explainer
        explainer = shap.Explainer(self.model, self.text_tfidf.toarray(), feature_names=self.vectoriser.get_feature_names_out())

        # compute SHAP values for the input
        shap_values = explainer(self.text_tfidf.toarray())

        # display the force plot
        # I can specify different classes using splicing
        shap_plot = shap.plots.waterfall(shap_values[:,:,0][0])
        return shap_plot

    

if __name__ == '__main__':
    print("This is the main file")
