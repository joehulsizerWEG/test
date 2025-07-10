# eight_utils.py
import os, datetime as dt, pandas as pd, numpy as np
import streamlit as st
from tools.eight.eight_work_api import (
    get_token, fetch_cdrs
)
from toolshed.tools.eight.eight_analytics import CallDataAnalyzer

CSV_NAME = "8x8_work_call_log_{start}_{end}.csv"

# ------------------------------------------------------------------ #
# 1.  Pull or load data – cached for 24 h
# ------------------------------------------------------------------ #
@st.cache_data(show_spinner="🔄  Loading 8x8 call data …", ttl=86_400)
def load_call_data(start: dt.date, end: dt.date) -> CallDataAnalyzer:
    fname = CSV_NAME.format(start=start, end=end)
    if os.path.exists(fname):
        df = pd.read_csv(fname)
    else:
        token = get_token()
        rows  = fetch_cdrs(token,
                           dt.datetime.combine(start, dt.time.min),
                           dt.datetime.combine(end,   dt.time.max))
        df = pd.DataFrame(rows)
        df.to_csv(fname, index=False)

    return CallDataAnalyzer(df)

# ------------------------------------------------------------------ #
# 2.  Sidebar filters – returns an analyzer scoped to the filters
# ------------------------------------------------------------------ #
def sidebar_filters():
    st.sidebar.header("📅 Filters")

    # ---- dates ----------------------------------------------------- #
    today = dt.date.today()
    start = st.sidebar.date_input("Start date", today.replace(day=1))
    end   = st.sidebar.date_input("End date", today)

    # ---- aggregation level ---------------------------------------- #
    agg_level = st.sidebar.radio(
        "Aggregation",
        ("Daily", "Weekly", "Monthly"),
        horizontal=True,
    )

    # ---- Load data ------------------------------------------------- #
    analyzer = load_call_data(start, end)

    # ---- branch / direction filters ------------------------------- #
    branches = sorted(analyzer.df["branch_key"].unique())
    dirs     = sorted(analyzer.df["direction"].str.capitalize().unique())

    sel_branches = st.sidebar.multiselect(
        "Branch", branches, default=branches)
    sel_dirs = st.sidebar.multiselect(
        "Direction", dirs, default=dirs)

    # ---- apply the filters to the raw DF --------------------------- #
    mask = analyzer.df["branch_key"].isin(sel_branches) & \
           analyzer.df["direction"].str.capitalize().isin(sel_dirs)
    analyzer.df = analyzer.df.loc[mask].copy()

    # ---- write filtered rows to session (optional) ---------------- #
    st.session_state["analyzer"]   = analyzer
    st.session_state["agg_level"]  = agg_level
    st.session_state["start_date"] = start
    st.session_state["end_date"]   = end
