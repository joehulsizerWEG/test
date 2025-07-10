import streamlit as st, pandas as pd
from toolshed.tools.eight.eight_utils import sidebar_filters

sidebar_filters()
analyzer   = st.session_state["analyzer"]
agg_level  = st.session_state["agg_level"]

st.subheader("Employees sorted by total minutes")
emp = analyzer.summarize().sort_values("total_minutes", ascending=False)
st.dataframe(emp, use_container_width=True, hide_index=True)

# --------------------------------------------------------------- #
# 2.  Calls‑per‑day grid
# --------------------------------------------------------------- #
freq = {"Daily":"D","Weekly":"W-MON","Monthly":"MS"}[agg_level]
df = analyzer.df.copy()
df["period"] = pd.to_datetime(df["startTimeUTC"]).dt.tz_convert("US/Eastern").dt.floor(freq)

# total & directional counts per employee per period
gp = (
    df.groupby(["emp_number","emp_name","period","direction_lc"])
      .agg(calls=("callId","count"), talk_min=("minutes","sum"))
      .reset_index()
)
pivot_calls = (
    gp.pivot_table(index=["emp_number","emp_name"], columns="direction_lc",
                   values="calls", aggfunc="mean")
      .rename(columns=str.capitalize)
)
pivot_calls["Total"] = pivot_calls.sum(axis=1)

pivot_talk = (
    gp.pivot_table(index=["emp_number","emp_name"], values="talk_min",
                   aggfunc="mean")
      .rename(columns={"talk_min":"Avg Talk Min"})
)

tbl = pivot_calls.join(pivot_talk)
st.subheader(f"Average per‑{agg_level.lower()} metrics")
st.dataframe(tbl.round(2).sort_values("Total",ascending=False),
             use_container_width=True)
