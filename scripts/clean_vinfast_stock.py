import joblib
import pandas as pd

class column_cleaner:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def clean_columns(self, ignore_cols: list = ["target"]):
        for column in self.data.columns:
            if column in ignore_cols:
                continue

            # Load the median, upper bound, and lower bound for this column
            try:
                median = joblib.load(f"../models/time_series/vinfast/median_lower_upper/{column}_median.pkl")
                upper  = joblib.load(f"../models/time_series/vinfast/median_lower_upper/{column}_upper.pkl")
                lower  = joblib.load(f"../models/time_series/vinfast/median_lower_upper/{column}_lower.pkl")
            except FileNotFoundError as e:
                print(f"Warning: Could not load scaler for '{column}': {e}")
                continue

            # Clip outliers to the stored median
            print(f"Imputing '{column}'...")
            self.data[column] = [
                median if (value < lower or value > upper) else value
                for value in self.data[column]
            ]
            print(f"  Done — column '{column}' imputed.")

        print(self.data)
        return self.data