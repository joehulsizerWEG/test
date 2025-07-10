import streamlit as st
from toolshed.tools.eight.eight_utils import sidebar_filters
import plotly.graph_objects as go

sidebar_filters()
analyzer = st.session_state["analyzer"]

st.subheader("Performance by Day & Hour")

# ------------------------------------------------------------------ #
# helper to render any heat‑map from a pivot DataFrame
# ------------------------------------------------------------------ #
def plot_heat(df, title, fmt=",.0f", colorscale="Blues"):
    fig = go.Figure(
        go.Heatmap(
            z=df.values,
            x=df.columns,
            y=df.index,
            text=df.round(2).applymap(lambda x: format(x, fmt)).values,
            texttemplate="%{text}",
            colorscale=colorscale,
            colorbar=dict(tickformat=fmt),
        )
    )
    fig.update_layout(title=title,
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ #
# 1.  Call count
# ------------------------------------------------------------------ #
pivot_cnt = analyzer.temporal_grid("call_count")
plot_heat(pivot_cnt, "Call count", " ,.0f", "Greens")

# ------------------------------------------------------------------ #
# 2.  Outbound answer %
# ------------------------------------------------------------------ #
pivot_ans_out = analyzer.temporal_grid("answer_rate_out", outbound=True)
plot_heat(pivot_ans_out, "Outbound answer %", ".1f", "Blues")

# ------------------------------------------------------------------ #
# 3.  Inbound answer %
# ------------------------------------------------------------------ #
pivot_ans_in = analyzer.temporal_grid("answer_rate_in", inbound=True)
plot_heat(pivot_ans_in, "Inbound answer %", ".1f", "OrRd")

# ------------------------------------------------------------------ #
# 4.  Avg talk minutes
# ------------------------------------------------------------------ #
pivot_min = analyzer.temporal_grid("avg_minutes")
plot_heat(pivot_min, "Average talk (min)", ".2f", "Purples")
