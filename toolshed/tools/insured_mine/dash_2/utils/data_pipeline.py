"""
utils.data_pipeline
————————
Keeps the heavy `InsuredMindWorker` class in its own
module and exposes *lightweight read‑only helpers* that pages import.

This prevents every page reload from re‑parsing Excel/CSV files.
"""

from __future__ import annotations

import datetime as _dt
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from tools.insured_mine.insured_mine_data import InsuredMindWorker  # <— your full original class


@lru_cache(maxsize=1)
def _worker() -> InsuredMindWorker:
    """Build once per interpreter (≈ once per gunicorn worker)."""
    file_date = _dt.date(2025, 6, 25)
    return InsuredMindWorker(
        card_file_name=f"C:/users/Luiso/data/insured_mine/Dealcards_{file_date:%m_%d_%Y}.xlsx",
        account_file_name=f"C:/users/Luiso/data/insured_mine/Accounts_{file_date:%m_%d_%Y}.csv",
        segment_file_name="C:/users/Luiso/data/census/SegmentsAll_20180809.csv",
    )


@lru_cache(maxsize=1)
def get_daily_kpi() -> pd.DataFrame:
    """Pre‑compute the three core KPIs used on the overview page."""
    df = _worker().data
    daily = (
        df.groupby("creation_date")
        .agg(
            leads=("TL_id", "size"),
            quotes=("Quoted Date", lambda s: s.notna().sum()),
            wins=("is_won", "sum"),
            median_mins=("CreationToFirstAction_min", "median"),
        )
        .sort_index()
        .reset_index()
        .rename(columns={"creation_date": "date"})
    )
    daily["win_rate"] = (
        daily["wins"].rolling(7, min_periods=1).sum()
        / daily["quotes"].rolling(7, min_periods=1).sum()
    )
    daily["median_mins_roll"] = daily["median_mins"].rolling(7, min_periods=1).median()
    return daily
