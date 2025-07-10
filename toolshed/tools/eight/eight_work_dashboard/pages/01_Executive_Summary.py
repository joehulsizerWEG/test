import streamlit as st, pandas as pd, plotly.express as px
from toolshed.tools.eight.eight_utils import sidebar_filters

sidebar_filters()
analyzer   = st.session_state["analyzer"]
agg_level  = st.session_state["agg_level"]

# --------------------------------------------------------------- #
# 1.  KPI tiles
# --------------------------------------------------------------- #
summary_br = analyzer.summarize_branch()
tot_calls  = summary_br["total_calls"].sum()
in_calls   = summary_br["total_inbound_calls"].sum()
out_calls  = summary_br["total_outbound_calls"].sum()
in_ans     = summary_br["inbound_answer_rate_pct"].mean()
out_ans    = summary_br["outbound_answer_rate_pct"].mean()
avg_mins   = analyzer.df["minutes"].mean()

kpi_cols = st.columns(6)
kpi_cols[0].metric("Total calls",      f"{tot_calls:,}")
kpi_cols[1].metric("Inbound calls",    f"{in_calls:,}")
kpi_cols[2].metric("Outbound calls",   f"{out_calls:,}")
kpi_cols[3].metric("Inbound answer %", f"{in_ans:,.1f}%")
kpi_cols[4].metric("Outbound answer %",f"{out_ans:,.1f}%")
kpi_cols[5].metric("Avg talk (min)",   f"{avg_mins:,.2f}")

# --------------------------------------------------------------- #
# 2.  Stacked‑bar – minutes per period
# --------------------------------------------------------------- #
freq = {"Daily":"D","Weekly":"W-MON","Monthly":"MS"}[agg_level]
gp = (
    analyzer.df
        .assign(period=lambda d: pd.to_datetime(d["startTimeUTC"])
                                 .dt.tz_convert("US/Eastern")
                                 .dt.floor(freq))
        .groupby(["period","direction_lc"])
        .agg(minutes=("minutes","sum"))
        .reset_index()
)

fig = px.bar(
    gp, x="period", y="minutes", color="direction_lc",
    labels={"period":agg_level, "minutes":"Minutes on phone",
            "direction_lc":"Direction"},
    title=f"Talk‑time by {agg_level.lower()}",
)
fig.update_layout(xaxis_title="")
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------- #
# 3.  Branch table
# --------------------------------------------------------------- #
st.subheader("Branch performance")
st.dataframe(
    summary_br.sort_values("total_minutes", ascending=False),
    use_container_width=True, hide_index=True,
)
