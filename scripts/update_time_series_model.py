import pandas as pd 
import numpy as np 

from xgboost import XGBRegressor 
import streamlit as st 
import os 
import sys 
import joblib 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from xgboost import XGBRegressor 

# metrics 
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error




BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# collecting the eod data 
#
#
# loading the eod data 
from vinfast_data_collection import load_eod_data 

# importing the cleaning dataset 
from clean_time_series import clean_iqr, clean_isolation_forest

"""
open_clf  = joblib.load(os.path.join(BASE_DIR, "models/time_series/vinfast/vinfast_open_forecaster.pkl"))
high_clf  = joblib.load(os.path.join(BASE_DIR, "models/time_series/vinfast/vinfast_high_forecaster.pkl"))
low_clf   = joblib.load(os.path.join(BASE_DIR, "models/time_series/vinfast/vinfast_low_forecaster.pkl"))
close_clf = joblib.load(os.path.join(BASE_DIR, "models/time_series/vinfast/vinfast_adjusted_close_forecaster.pkl"))
"""

df = pd.read_csv(f"{BASE_DIR}/data/vinfast_data_cleaned.csv") 
df['date'] = pd.to_datetime(df['date']) 

# removing volume
df.drop(['volume', 'close'],axis=1,inplace=True)


# sorting the dataframe by date
df = df.sort_values(['date'],ascending=True) 



def clean_dataframe(df: pd.DataFrame): 
    # loading hte isolation_forest module 
    print ("Applying the Isolation Forest...")
    clean_df = clean_isolation_forest(df=df) 

    clean_df.fit_transform() 

    print ("Lagging the dataset...")
    cleaned_df_lagged = clean_df.build_lagging() 

    return cleaned_df_lagged









def split_train_test(df: pd.DataFrame, target_column): 

    # assuming the data is already sorted 




    # splitting into X_train and X_test 
    print ("Splitting the data...")
    df_length = len(df) 
    print (f"Length of df: {df_length}")
    train_size = round(df_length * 0.8)
    print (f"Train size: {train_size}")
    test_size = round(df_length * 0.2)
    print (f"Test size: {test_size}")
    print ("------------------------")



    print ("Creating X...")
    X = df.drop([f"{target_column}"], axis=1)
    print (f"X: {X}") 
    X_train = X.iloc[0:train_size]
    print (f"X_train: {X_train.head()}")
    X_test = X.iloc[train_size:df_length] 
    print (f"X_test: {X_test.head()}")

    print ("Creating Y...")
    y = df[f"{target_column}"]
    print (f"y: {y[0:10]}")
    Y_train = y.head(train_size).values.reshape(-1,1) 
    print (f"Y_train: {Y_train[0:10]}")
    Y_test = y.tail(test_size).values.reshape(-1,1) 
    print (f"Y_test: {Y_test[0:10]}")


    # returning hte data 
    return X, y, X_train, X_test, Y_train, Y_test 




def build_model(X_train: pd.DataFrame, Y_train, X_test: pd.DataFrame):
    
    # using XGBoost 



    model = XGBRegressor() 
    model.fit(X_train, Y_train) 
    pred = model.predict(X_test) 

    # optimising 


    return model, pred  


def generate_metrics(model, X_train, X_test, Y_train, Y_test, X, y): 
    print ("test scores...")
    print ("------------------\n\n")
    print (cross_val_score(estimator=model, X=X, y=y, cv=4, scoring='r2'))


    print ("Done\n")



# training models and cleaning 






def main(): 
    print ("Loading the script...")
    cleaned_df = clean_dataframe(df=df) 

    print ("Train/Test Splitting...") 
    """
    split_datasets is a dictionary containing target_column as the key and (X_train, Y_train,... etc) as 
    values

    """

    for target_column in cleaned_df.columns: 
        # not using volume 
        if target_column != 'day' and target_column != 'month' and target_column != 'year' and target_column != 'close':
            X, y, X_train, X_test, Y_train, Y_test = split_train_test(df=cleaned_df, target_column=target_column)
            # I can store the data into a dictionary for linear time search 
            
            print ("Training the model...") 

            model, pred = build_model(X_train=X_train, Y_train=Y_train, X_test=X_test)




            print ("Generating metrics...") 
            print (f"Target column: {target_column}")
            print (generate_metrics(model, X_train, X_test, Y_train, Y_test, X, y))
            joblib.dump(model, f"{BASE_DIR}/models/time_series/vinfast/vinfast_{target_column}_forecaster.pkl")

    print ("Done")
    # the format for "models" is {"model": [model, pred, ]}



        


if __name__ == "__main__": 
    print (main()) 
