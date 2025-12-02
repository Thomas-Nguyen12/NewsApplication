import pandas as pd 
import joblib 
import numpy as np
import shap 




class classifier: 
    def __init__(self, text:str): 
        self.text = pd.DataFrame({'text': [text]}) 
        
        self.model = joblib.load("data/news_classifier.pkl") 
        print (self.model)
        self.vectoriser = joblib.load("data/tfidf_vectoriser.pkl")
        print (self.vectoriser)
        self.topic_hashmap = joblib.load("data/topic_hashmap.pkl")
        print (self.topic_hashmap)
        # preprocessing the text
        # removing unecessary characters
        self.text['text'] = self.text['text'].str.replace("[^a-zA-Z0-9]+", " ", regex=True)
        self.text['text'] = self.text['text'].str.strip() 
        
        
        
       
        self.text_tfidf = self.vectoriser.transform(self.text['text']) 
        print (self.text_tfidf)
        print ("This is a binary relevance xgboost classifier (fits multiple binary classifier models)")
        
        
    def predict(self): 
        prediction = self.model.predict(self.text_tfidf)
        prediction = prediction.toarray()
        print (prediction)
        
        # I need to assign the prediction to the feature names
        print ("Creating dataframe...")
        prediction_df = pd.DataFrame(prediction, columns=self.topic_hashmap.keys())
        
        print (prediction_df)

        # returning the output
        print ("Filtering...")
        # Maybe I can format the results so that only columns with a one in it are returned
        prediction_df_filtered = prediction_df.loc[:, (prediction_df == 1).all()]

        return prediction_df_filtered.columns 
    
    def predict_proba(self): 
        
        predict_proba = self.model.predict_proba(self.text_tfidf) 
        predict_proba = predict_proba.toarray()
        predict_proba_df = pd.DataFrame(predict_proba, columns=self.topic_hashmap.keys())
        predict_proba_df_filtered = predict_proba_df.loc[:, (predict_proba_df >= 0.5).all()]
        return predict_proba_df_filtered.values
    
    def explain(self): 
        # implementing shap 
        
        explainer = shap.Explainer(self.model,feature_names=self.vectoriser.get_feature_names_out())

        # compute SHAP values for the input
        
        shap_values = explainer(self.text_tfidf.toarray())
    
    
        # Here, I can splice the shap_values for each feature
        #shap_force_plot = shap.force_plot(shap_values)
        
        # summary plots will cover all classes
        shap_summary_plot = shap.summary_plot(shap_values)
        return shap_summary_plot

if __name__ == "__main__":
    print ("This is the main file")