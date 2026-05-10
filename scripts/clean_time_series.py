import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
import scipy.stats as stats 



from sklearn.ensemble import IsolationForest

# Prepare the data



# This script will remove anomalies
# I can use isolation forest or iqr
# I can also use the "target variable" as an additional parameter
# the target variable should NOT be cleaned


"""
high_anomalies = df[['high']].copy()

# Fit Isolation Forest
iso_forest = IsolationForest(
    contamination='auto',  # Expected proportion of anomalies (5%)
    random_state=42,
    n_estimators=100
)

df['anomaly'] = iso_forest.fit_predict(high_anomalies)
df['anomaly_score'] = iso_forest.decision_function(high_anomalies)

# -1 = anomaly, 1 = normal
df['is_anomaly'] = df['anomaly'] == -1

# View anomalies
anomalies = df[df['is_anomaly']][['date', 'high', 'anomaly_score']]
print(f"Found {len(anomalies)} anomalies out of {len(df)} records")
print(anomalies)
"""

print ("This script is based on the idea that you wish to use lagging stock values to predict future stock values")
print ("Make sure you have cleaned your data first") 
print ("This script is for preprocessing for modelling...")
print ("The date should be within the index")
# This is an unsupervised technique
class clean_isolation_forest:

    def __init__(self, df: pd.DataFrame) -> pd.DataFrame: 

        self.df = df
        self.df.columns = self.df.columns.str.lower()




        # doing some preemptive formatting
        if 'date' in self.df.columns:

            self.df.index = pd.to_datetime(df['date']) 
            self.df.drop(['date'],axis=1,inplace=True)


            self.df['day'] = self.df.index.day
            self.df['month'] = self.df.index.month 
            self.df['year'] = self.df.index.year

        # checking if the date is in the index
        elif self.df.index == pd.to_datetime(self.df.index): 
            self.df.index = pd.to_datetime(self.df.index)
            print ("The datetime is in the index... moving on...")
        
        else:
            print ("Please check the formatting of the date column")



    def fit_transform(self) -> pd.DataFrame: 
        # this action will be performed on self.df
        for column in self.df.columns: 
                if column != "day" and column != "month" and column != "year":
                    median = self.df[column].median() 
                    column_anomalies = self.df[column].copy() 


                    iso_forest = IsolationForest(
                        contamination='auto',  # Expected proportion of anomalies (5%)
                        random_state=42,
                        n_estimators=100
                    )

                    self.df[f'{column}_anomaly'] = iso_forest.fit_predict(column_anomalies)
                    self.df[f'{column}_anomaly_score'] = iso_forest.decision_function(column_anomalies)

                    # -1 = anomaly, 1 = normal
                    self.df[f'{column}_is_anomaly'] = self.df[f'{column}_anomaly'] == -1

                    # View anomalies
                    anomalies = self.df[self.df['is_anomaly']][['date', 'high', f'{column}_anomaly_score']]
                    print(f"Found {len(anomalies)} anomalies out of {len(self.df)} records")
                    print(anomalies)
                    
                    # imputing with median 
                    print ("Imputing with median...")
                    self.df[f"{column}_isolation_forest_cleaned"] = [median if anomaly == True else anomaly for anomaly in self.df[f'{column}_is_anomaly']]
                else:
                    pass

        result = self.df



        return result
    
    def build_lagging(self, dataframe: pd.DataFrame) -> pd.DataFrame:

    # building the lagging
        print ("Created the lagged values...")
        for column in dataframe.columns: 

            if column != 'year' and column != 'month' and column !='day': 
                """
                lagged_volume = X_train['volume_imputed'].values
                lagged_volume = np.insert(lagged_volume, 0, 0)
                lagged_volume = lagged_volume[:-1]
                """
                lagged_values = dataframe[column].values
                lagged_values = np.insert(lagged_values, 0, 0) 
                lagged_values = lagged_values[:-1]

                dataframe[f"lagged_{column}"] = lagged_values
        # removing the first row
        print ("Removing the first row")

        result = dataframe.iloc[1:]
        return result



# This is a supervised technique
class clean_iqr:
    def __init__(self, df: pd.DataFrame) -> pd.DataFrame:
        self.df = df


        self.column_iqr_median_upper_lower = []


        # cleaning some of the data

        
        self.df.index = pd.to_datetime(df['date']) 
        self.df.drop(['date'],axis=1,inplace=True)
        self.df['day'] = self.df.index.day
        self.df['month'] = self.df.index.month 
        self.df['year'] = self.df.index.year

    
    def fit(self) -> pd.DataFrame: 
        # This just calculates the median, lower and upper
        # This should just be used for X_train
        print ("Calculating the upper, lower, median and iqr...")

        for column in self.df.columns: 
            if column != 'day' and column != 'month' and column != 'year':
                print (f"Analysing column: {column}")
                iqr = stats.iqr(self.df[column])
                upper = np.percentile(self.df[column], 75) + (1.5 * iqr)
                lower = np.percentile(self.df[column], 25) - (1.5 * iqr)
                median = self.df[column].median()

                # I should return this as a dictionary so I can access the values
                self.column_iqr_median_upper_lower.append((column, iqr, median, upper, lower))

        return pd.DataFrame(self.column_iqr_median_upper_lower, columns=['column', 'iqr', 'median', 'upper', 'lower']) 


    def fit_transform(self) -> pd.DataFrame: 
        # This should be used for X_train
        print ("Using the 1.5 * IQR method for anomaly detection...")

        # using the iqr method
        for column in self.df.columns: 
            if column != 'day' and column != 'month' and column != 'year':
                print (f"Analysing column: {column}")
                iqr = stats.iqr(self.df[column])
                upper = np.percentile(self.df[column], 75) + (1.5 * iqr)
                lower = np.percentile(self.df[column], 25) - (1.5 * iqr)
                median = self.df[column].median()
                self.df[f"{column}_iqr_imputed"] = [median if value > upper or value < lower else value for value in self.df[column]]
                

            else:
                print (f"`{column}` is a datetime. Ignoring this...")
        return self.df
        
    def transform(self, column_stats: pd.DataFrame, dataframe: pd.DataFrame) -> pd.DataFrame: 
        # This should be used for X_test
        # the stats should be in self.columns_iqr_median_Upper_lower
        if 'date' in dataframe.columns:
            dataframe.index = pd.to_datetime(dataframe['date'])
            dataframe['day'] = dataframe.index.day
            dataframe['month'] = dataframe.index.month
            dataframe['year'] = dataframe.index.year
            dataframe.drop(['date'],axis=1,inplace=True)
     
        if column_stats is not None:


            for column in dataframe.columns: 

                if column != "month" and column != "day" and column != "year": 
                    


                    # I need to extract the values for column_iqr_median_upper_lower
                    print ("Transforming... Dataframe") 
                    median = column_stats[column_stats['column'] == column]['median'].values
                    print (f"median: {median}")
                    upper = column_stats[column_stats['column'] == column]['upper'].values
                    print (f"upper: {upper}")
                    lower = column_stats[column_stats['column'] == column]['lower'].values
                    print (f"lower: {lower}")
                    
                    dataframe[f"{column}_iqr_imputed"] = [median if value > upper or value < lower else value for value in dataframe[column]]
            
        else:
            return "Please fit the cleaner first on your X_train and then use this method to clean your X_test"

        # dropping the irrelevant columns
      


        return dataframe
    
    def build_lagging(self, dataframe: pd.DataFrame) -> pd.DataFrame:

        # building the lagging
        print ("Created the lagged values...")
        for column in dataframe.columns: 

            if column != 'year' and column != 'month' and column !='day': 
                """
                lagged_volume = X_train['volume_imputed'].values
                lagged_volume = np.insert(lagged_volume, 0, 0)
                lagged_volume = lagged_volume[:-1]
                """
                lagged_values = dataframe[column].values
                lagged_values = np.insert(lagged_values, 0, 0) 
                lagged_values = lagged_values[:-1]

                dataframe[f"lagged_{column}"] = lagged_values
        # removing the first row
        print ("Removing the first row")

        result = dataframe.iloc[1:]
        return result
    