import pandas as pd
import joblib
import numpy as np
import shap


class ai_detector:
    def __init__(self, text:str):

        # store input text
        self.text = pd.DataFrame({'text': [text]})

        # load TF-IDF vectorizer
        self.vectoriser = joblib.load("ai_vectoriser.pkl")
        
        # removing unecessary special characters
        self.text['text'] = self.text['text'].str.replace("[^a-zA-Z0-9]+", " ", regex=True).str.strip()

        # transform text
        self.text_tfidf = self.vectoriser.transform(self.text['text'])

        # load model
        self.model = joblib.load("ai_detector.pkl")



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
        explainer = shap.Explainer(self.model, feature_names=self.vectoriser.get_feature_names_out())

        # compute SHAP values for the input
        shap_values = explainer(self.text_tfidf.toarray())

        # display the force plot
        shap_plot = shap.plots.waterfall(shap_values[:,:,0][0])
        return shap_plot

if __name__ == '__main__':
    print("This is the main file")
