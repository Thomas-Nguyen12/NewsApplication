import pandas as pd
import joblib
import numpy as np
import shap
import re
vectoriser = joblib.load("models/ai_detector/ai_vectoriser.pkl")
model = joblib.load("models/ai_detector/ai_detector.pkl")

class classifier:
    def __init__(self, text:str):

        # store input text
        self.text = re.sub("[^a-zA-Z0-9]+", " ", text).strip()
        

        # load TF-IDF vectorizer
        
        
        # removing unecessary special characters
       
        
        

        # transform text
        self.text_list = [self.text]
        self.text_tfidf = vectoriser.transform(self.text_list)

        # load model
        



    def predict(self):
        prediction = model.predict(self.text_tfidf)
        if prediction == 0: 
            return "This is likely a human generated text"
        
        
        else:
            return "This is likely an AI generated text"


    def predict_proba(self):
        prediction_proba = model.predict_proba(self.text_tfidf)
        prediction_probabilities = pd.DataFrame(prediction_proba, columns=['human_generated', 'AI_generated'])
        prediction_probabilities_filtered = prediction_probabilities.loc[:, (prediction_probabilities >= 0.5).all()]
        return prediction_probabilities_filtered.values[0][0]


    def explain(self):
        # create SHAP explainer
        feature_names=vectoriser.get_feature_names_out()
        input_df = pd.DataFrame(self.text_tfidf.toarray(), columns=feature_names)
        explainer = shap.TreeExplainer(model)

        # compute SHAP values for the input




        shap_values = explainer(input_df)

        # display the force plot
        # I can specify different classes using splicing
        prediction = model.predict(self.text_tfidf)[0]
        if prediction == 0:

            shap_plot = shap.plots.waterfall(shap_values[:,:,0][0])
        else:
            shap_plot = shap.plots.waterfall(shap_values[:,:,1][0])
        return shap_plot



if __name__ == '__main__':
    print("This is the main file")
