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


# This is an unsupervised technique
class clean_isolation_forest:

    def __init__(self, df: pd.DataFrame, target: pd.Series) -> pd.DataFrame: 

        self.df = df
        self.target = target

        # doing some preemptive formatting
       
        


    def fit_transform(self) -> pd.DataFrame: 
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


        return self.df


# This is a supervised technique
class clean_iqr:
    def __init__(self, df: pd.DataFrame) -> pd.DataFrame:
        self.df = df

    def fit_transform(self) -> pd.DataFrame: 
        print ("Using the 1.5 * IQR method for anomaly detection...")
        column_iqr_median_upper_lower = []
        # using the iqr method
        for column in self.df.columns: 
            if column != 'day' and column != 'month' and column != 'year' and column != 'date': 
                print (f"Analysing column: {column}")
                iqr = stats.iqr(self.df[column])
                upper = np.percentile(self.df[column], 75) + (1.5 * iqr)
                lower = np.percentile(self.df[column], 25) - (1.5 * iqr)
                median = self.df[column].median()
                self.df[f"{column}_iqr_imputed"] = [median if value > upper or value < lower else value for value in self.df[column]]
                column_iqr_median_upper_lower.append((column, iqr,median,upper,lower))

            else:
                print (f"`{column}` is a datetime. Ignoring this...")

        return self.df, column_iqr_median_upper_lower


    def remove_anomalies(self): 
        # I will impute the anomalies with the median 

        pass 
