"""
Edge case tests for pandas_market_calendars usage in the VinFast (VFS)
forecasting pipeline.

These guard against the class of bug already hit once in this project:
a UTC/NASDAQ timezone mismatch that caused incorrect trading-day
alignment in the fan-chart forecaster.
"""

import pandas as pd
import pandas_market_calendars as mcal
import pytest

nasdaq = mcal.get_calendar('NASDAQ')


def _schedule(start, end):
    return nasdaq.schedule(start_date=start, end_date=end)


def test_known_holiday_is_excluded():
    """New Year's Day 2026 should not appear as a trading day."""
    schedule = _schedule('2026-01-01', '2026-01-02')
    trading_days = schedule.index.date
    assert pd.Timestamp('2026-01-01').date() not in trading_days


def test_known_trading_day_is_included():
    """A plain midweek day with no holiday should be a trading day."""
    schedule = _schedule('2026-03-10', '2026-03-11')
    trading_days = schedule.index.date
    assert pd.Timestamp('2026-03-10').date() in trading_days


def test_thanksgiving_early_close():
    """Day after Thanksgiving is a scheduled early close (1:00 PM ET)."""
    schedule = _schedule('2026-11-27', '2026-11-27')
    assert not schedule.empty, "Expected a trading day, market was not closed"

    market_close = schedule.iloc[0]['market_close']
    # Early close should be 13:00 ET -> 18:00 UTC (standard time in Nov)
    assert market_close.tz_convert('America/New_York').hour == 13


def test_market_close_timezone_is_consistent():
    """
    Regression test for the UTC/NASDAQ mismatch bug: market_close must
    convert to the expected Eastern time regardless of the timezone the
    schedule DataFrame is stored/compared in.
    """
    schedule = _schedule('2026-03-10', '2026-03-10')
    close_utc = schedule.iloc[0]['market_close']
    close_et = close_utc.tz_convert('America/New_York')

    assert close_et.hour == 16  # standard 4:00 PM ET close
    assert close_utc.tzinfo is not None, "market_close must be tz-aware"


def test_weekend_has_no_schedule():
    """Saturdays should produce an empty schedule."""
    schedule = _schedule('2026-03-14', '2026-03-14')  # a Saturday
    assert schedule.empty


def test_forecast_horizon_skips_holidays():
    """
    Simulates generating N future trading days for the fan-chart
    forecaster and checks that known holidays are never included,
    even when they fall inside the requested date range.
    """
    schedule = _schedule('2026-12-20', '2027-01-05')
    trading_days = set(schedule.index.date)

    christmas = pd.Timestamp('2026-12-25').date()
    new_years = pd.Timestamp('2027-01-01').date()

    assert christmas not in trading_days
    assert new_years not in trading_days


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
