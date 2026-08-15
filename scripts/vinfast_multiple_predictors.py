import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import joblib
import pandas_market_calendars as mcal
from datetime import timedelta, date
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = f"{BASE_DIR}/models/time_series/vinfast"

open_clf            = joblib.load(f"{MODEL_DIR}/vinfast_open_isolation_forest_cleaned_forecaster.pkl")
high_clf            = joblib.load(f"{MODEL_DIR}/vinfast_high_isolation_forest_cleaned_forecaster.pkl")
low_clf             = joblib.load(f"{MODEL_DIR}/vinfast_low_isolation_forest_cleaned_forecaster.pkl")
adjusted_close_clf  = joblib.load(f"{MODEL_DIR}/vinfast_adjusted_close_isolation_forest_cleaned_forecaster.pkl")

nasdaq = mcal.get_calendar("NASDAQ")

# The four price columns your training script now produces one model for
# each of (raw close excluded entirely — adjusted_close is the sole close-type
# feature/target) — every model's features are "all of these except itself",
# matching `X = cleaned_df.drop([target_column], axis=1)` in the training
# script exactly.
PRICE_COLUMNS = [
    "open_isolation_forest_cleaned",
    "high_isolation_forest_cleaned",
    "low_isolation_forest_cleaned",
    "adjusted_close_isolation_forest_cleaned",
]

BASE_COLUMNS = ["day", "month", "year"]


class forecaster:
    """
    Iterative multi-step price forecaster for VinFast (VFS).

    Each day's predicted O/H/L/Close/Adj-Close becomes the lagged features
    for the next day, allowing n-period ahead forecasts from a single seed
    observation.

    Matches the training script exactly: four separate models (open, high,
    low, adjusted_close — raw close excluded entirely), each trained with
    every other price column (plus day/month/year) as features and only its
    own column excluded — e.g. open_clf's features are [day, month, year,
    high, low, adjusted_close].

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
    seed_adjusted_close : float
        Actual adjusted close price on input_date.
    """

    def __init__(
        self,
        input_date,
        n_periods: int,
        seed_open: float,
        seed_high: float,
        seed_low: float,
        seed_adjusted_close: float,
    ):
        self.input_date  = pd.Timestamp(input_date)
        self.n_periods   = n_periods
        self.end_date    = self.input_date + timedelta(days=n_periods * 2)

        self.seed_values = {
            "open_isolation_forest_cleaned":            seed_open,
            "high_isolation_forest_cleaned":             seed_high,
            "low_isolation_forest_cleaned":              seed_low,
            "adjusted_close_isolation_forest_cleaned":   seed_adjusted_close,
        }

        self._results: pd.DataFrame | None = None

        # Pre-compute valid trading days from input_date up to a safe upper bound
        schedule = nasdaq.schedule(
            start_date=self.input_date,
            end_date=self.input_date + timedelta(days=n_periods * 2 + 30),
        )
        self._trading_days = mcal.date_range(schedule, frequency="1D")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_trading_day(self, current: pd.Timestamp) -> pd.Timestamp:
        """Return the next NASDAQ trading day after current, skipping weekends and holidays."""
        current_utc = current.normalize().tz_localize("UTC") + timedelta(days=1)
        future = self._trading_days[self._trading_days >= current_utc]
        if future.empty:
            raise ValueError(
                f"No trading days found after {current.date()}. "
                "Extend the schedule end_date in __init__."
            )
        return future[0].normalize().tz_localize(None)

    @staticmethod
    def _build_features(day: int, month: int, year: int, lag_values: dict, exclude_col: str) -> pd.DataFrame:
        """
        Build the feature row for one model: day/month/year plus every
        price column except `exclude_col` (that model's own target),
        matching `X = cleaned_df.drop([target_column], axis=1)`.
        """
        row = {"day": day, "month": month, "year": year}
        for col in PRICE_COLUMNS:
            if col != exclude_col:
                row[col] = lag_values[col]

        ordered_columns = BASE_COLUMNS + [c for c in PRICE_COLUMNS if c != exclude_col]
        return pd.DataFrame([row], columns=ordered_columns)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forecast(self) -> pd.DataFrame:
        """
        Iteratively predict O/H/L/Close/Adj-Close for the next `n_periods`
        trading days.

        Returns
        -------
        pd.DataFrame
            Columns: date, predicted_open, predicted_high, predicted_low,
                      predicted_close, predicted_adjusted_close
        """
        if self._results is not None:
            return self._results

        rows = []
        lag_values = dict(self.seed_values)

        current_date = self._next_trading_day(self.input_date)

        for _ in range(self.n_periods):
            day, month, year = current_date.day, current_date.month, current_date.year

            pred_open = float(open_clf.predict(
                self._build_features(day, month, year, lag_values, "open_isolation_forest_cleaned")
            )[0])
            pred_high = float(high_clf.predict(
                self._build_features(day, month, year, lag_values, "high_isolation_forest_cleaned")
            )[0])
            pred_low = float(low_clf.predict(
                self._build_features(day, month, year, lag_values, "low_isolation_forest_cleaned")
            )[0])
            pred_adjusted_close = float(adjusted_close_clf.predict(
                self._build_features(day, month, year, lag_values, "adjusted_close_isolation_forest_cleaned")
            )[0])

            rows.append({
                "date":                     current_date.date(),
                "predicted_open":           pred_open,
                "predicted_high":           pred_high,
                "predicted_low":            pred_low,
                "predicted_adjusted_close": pred_adjusted_close,
            })

            # roll all four lags forward together
            lag_values = {
                "open_isolation_forest_cleaned":            pred_open,
                "high_isolation_forest_cleaned":             pred_high,
                "low_isolation_forest_cleaned":              pred_low,
                "adjusted_close_isolation_forest_cleaned":   pred_adjusted_close,
            }

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
        from scipy import stats  # NOTE: was missing in the original script

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

            t_crit             = stats.t.ppf(0.975, df=n - 1)
            margin             = t_crit * std / np.sqrt(n)
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
        """Quick fan-chart of all five predicted series."""
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
        input_date          = "2024-12-31",
        n_periods            = 30,
        seed_open            = 4.50,
        seed_high            = 4.75,
        seed_low             = 4.30,
        seed_adjusted_close  = 4.60,
    )

    predictions = fc.forecast()
    print(predictions.head())

    summary = fc.calculate_parameters()
    print("\nSummary statistics:")
    print(summary)

    fc.save("vinfast_30d_forecast.csv")
    fc.plot()
