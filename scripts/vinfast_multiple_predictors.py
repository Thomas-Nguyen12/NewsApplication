import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import sys
import joblib
from datetime import timedelta, date

open_clf  = joblib.load("../models/time_series/vinfast/vinfast_open_forecaster.pkl")
high_clf  = joblib.load("../models/time_series/vinfast/vinfast_high_forecaster.pkl")
low_clf   = joblib.load("../models/time_series/vinfast/vinfast_low_forecaster.pkl")
close_clf = joblib.load("../models/time_series/vinfast/vinfast_adjusted_close_forecaster.pkl")


class forecaster:
    """
    Iterative multi-step price forecaster for VinFast (VFS).

    Each day's predicted O/H/L/C becomes the lagged features for the next day,
    allowing n-period ahead forecasts from a single seed observation.

    Parameters
    ----------
    input_date : datetime-like
        The *last known* trading date (seed row).
    n_periods : int
        Number of trading days to forecast ahead.
    seed_open : float
        Actual open price on input_date.
    seed_high : float
        Actual high price on input_date.
    seed_low : float
        Actual low price on input_date.
    seed_close : float
        Actual adjusted close price on input_date.
    """

    def __init__(
        self,
        input_date,
        n_periods: int,
        seed_open: float,
        seed_high: float,
        seed_low: float,
        seed_close: float,
    ):
        self.input_date  = pd.Timestamp(input_date)
        self.n_periods   = n_periods          # keep as int — used for loop count
        self.end_date    = self.input_date + timedelta(days=n_periods * 2)  # rough upper bound
        self.seed_open   = seed_open
        self.seed_high   = seed_high
        self.seed_low    = seed_low
        self.seed_close  = seed_close
        self._results: pd.DataFrame | None = None   # cached after first call

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _next_trading_day(current: pd.Timestamp) -> pd.Timestamp:
        """Advance by one calendar day, skipping weekends."""
        nxt = current + timedelta(days=1)
        while nxt.weekday() >= 5:   # 5 = Saturday, 6 = Sunday
            nxt += timedelta(days=1)
        return nxt

    @staticmethod
    def _open_features(day: int, month: int, year: int,
                       lag_close: float, lag_high: float, lag_low: float) -> pd.DataFrame:
        """Feature row for open_clf (no IQR suffix)."""
        return pd.DataFrame([{
            "day":                  day,
            "month":                month,
            "year":                 year,
            "lagged_adjusted_close": lag_close,
            "lagged_high":          lag_high,
            "lagged_low":           lag_low,
        }])

    @staticmethod
    def _other_features(day: int, month: int, year: int,
                        lag_open: float, lag_high: float,
                        lag_low: float, lag_close: float) -> pd.DataFrame:
        """Feature row for high_clf / low_clf / close_clf (IQR-imputed names)."""
        return pd.DataFrame([{
            "day":                              day,
            "month":                            month,
            "year":                             year,
            "lagged_open_iqr_imputed":          lag_open,
            "lagged_high_iqr_imputed":          lag_high,
            "lagged_low_iqr_imputed":           lag_low,
            "lagged_adjusted_close_iqr_imputed": lag_close,
        }])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forecast(self) -> pd.DataFrame:
        """
        Iteratively predict O/H/L/C for the next `n_periods` trading days.

        Returns
        -------
        pd.DataFrame
            Columns: date, predicted_open, predicted_high,
                      predicted_low, predicted_adjusted_close
        """
        if self._results is not None:      # return cached result on repeat calls
            return self._results

        rows = []
        lag_open  = self.seed_open
        lag_high  = self.seed_high
        lag_low   = self.seed_low
        lag_close = self.seed_close

        current_date = self._next_trading_day(self.input_date)

        for _ in range(self.n_periods):
            day, month, year = current_date.day, current_date.month, current_date.year

            # --- open (uses its own feature schema) ---
            X_open  = self._open_features(day, month, year, lag_close, lag_high, lag_low)
            pred_open = float(open_clf.predict(X_open)[0])

            # --- high, low, close (share the IQR feature schema) ---
            X_other = self._other_features(day, month, year,
                                           pred_open, lag_high, lag_low, lag_close)
            pred_high  = float(high_clf.predict(X_other)[0])
            pred_low   = float(low_clf.predict(X_other)[0])
            pred_close = float(close_clf.predict(X_other)[0])

            rows.append({
                "date":                    current_date.date(),
                "predicted_open":          pred_open,
                "predicted_high":          pred_high,
                "predicted_low":           pred_low,
                "predicted_adjusted_close": pred_close,
            })

            # roll lags forward
            lag_open  = pred_open
            lag_high  = pred_high
            lag_low   = pred_low
            lag_close = pred_close

            current_date = self._next_trading_day(current_date)

        self._results = pd.DataFrame(rows)
        return self._results

    def calculate_parameters(self) -> pd.DataFrame:
        """
        Compute summary statistics for each predicted price series.

        Returns
        -------
        pd.DataFrame
            Index = price column name.
            Columns: mean, std, min, max, ci_lower_95, ci_upper_95
        """
        df = self.forecast()
        price_cols = [
            "predicted_open",
            "predicted_high",
            "predicted_low",
            "predicted_adjusted_close",
        ]

        summary_rows = []
        for col in price_cols:
            series = df[col]
            n      = len(series)
            mean   = series.mean()
            std    = series.std(ddof=1)

            # 95 % t-interval (robust for small n)
            t_crit         = stats.t.ppf(0.975, df=n - 1)
            margin         = t_crit * std / np.sqrt(n)
            ci_lower, ci_upper = mean - margin, mean + margin

            summary_rows.append({
                "series":      col,
                "mean":        round(mean, 4),
                "std":         round(std, 4),
                "min":         round(series.min(), 4),
                "max":         round(series.max(), 4),
                "ci_lower_95": round(ci_lower, 4),
                "ci_upper_95": round(ci_upper, 4),
            })

        return pd.DataFrame(summary_rows).set_index("series")

    def save(self, path: str = "vinfast_forecast.csv") -> None:
        """Save forecast DataFrame to CSV."""
        self.forecast().to_csv(path, index=False)
        print(f"Saved → {path}")

    def plot(self) -> None:
        """Quick fan-chart of all four predicted series."""
        df = self.forecast()
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.fill_between(df["date"], df["predicted_low"], df["predicted_high"],
                        alpha=0.15, label="Low–High band")
        for col, label in [
            ("predicted_open",           "Open"),
            ("predicted_adjusted_close", "Adj. Close"),
        ]:
            ax.plot(df["date"], df[col], label=label)

        ax.set_title("VinFast — iterative price forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fc = forecaster(
        input_date  = "2024-12-31",
        n_periods   = 30,
        seed_open   = 4.50,
        seed_high   = 4.75,
        seed_low    = 4.30,
        seed_close  = 4.60,
    )

    predictions = fc.forecast()
    print(predictions.head())

    summary = fc.calculate_parameters()
    print("\nSummary statistics:")
    print(summary)

    fc.save("vinfast_30d_forecast.csv")
    fc.plot()