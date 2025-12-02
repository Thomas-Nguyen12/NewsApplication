
import joblib
import shap
print ("Loading the data...") 
X_test_tfidf=joblib.load("X_test_tfidf.pkl") 
model=joblib.load("ai_detector.pkl") 
vectoriser = joblib.load("ai_vectoriser.pkl") 

print ("Setting up the explainer...") 
explainer = shap.Explainer(model, X_test_tfidf.toarray(), feature_names=vectoriser.get_feature_names_out()) 

print ("Explaining the test data...") 
shap_values = explainer(X_test_tfidf.toarray()) 
print ("Done. Saving the results...")
joblib.dump(shap_values, 'shap_values_X_test.pkl') 





if __name__ == '__main__':
    print ("Running...") 

