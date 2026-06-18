import joblib
import pandas as pd
import numpy
import re
import shap
import os


from nltk.corpus import stopwords
import spacy
# Instead of this:
# model = joblib.load("models/news_topic_classifier/news_topic_classifier.pkl")

# Use this:

stop_words = list(set(stopwords.words('english')))
nlp = spacy.load("en_core_web_sm")
def lemmatize(text):
    doc = nlp(text)
    # Turn it into tokens, ignoring the punctuation
    tokens = [token for token in doc if not token.is_punct]
    # Convert those tokens into lemmas, EXCEPT the pronouns, we'll keep those.
    lemmas = [token.lemma_ if token.pos_ != 'PRON' else token.orth_ for token in tokens]
    return lemmas


