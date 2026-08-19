import joblib 
import pandas as pd
import numpy as np
import re
import shap
import os
import spacy


import streamlit as st

from pathlib import Path 
import plotly.express as px
import sys 

BASE_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = (BASE_DIR / "scripts").resolve()


# this page will be for chat bots 
from langchain_openai.chat_models import ChatOpenAI

st.title("🦜🔗 ChatBot")
with st.sidebar:
    st.header("Navigation Bar")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")


def generate_response(input_text):
    model = ChatOpenAI(temperature=0.7, api_key=openai_api_key)
    st.info(model.invoke(input_text))


with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "What are the key times to invest in vinfast?",
    )
    submitted = st.form_submit_button("Submit")
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter your OpenAI API key!", icon="⚠")
    if submitted and openai_api_key.startswith("sk-"):
        generate_response(text)