"""
Leads / quotes / wins time‑series overview.
"""

from __future__ import annotations

import datetime as _dt

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import Input, Output, callback, html, dcc, register_page
from tools.insured_mine.dash_2.utils.theme import COLOR_PRIMARY
from tools.insured_mine.dash_2.utils.data_pipeline import get_daily_kpi  # thin helper, see below

register_page(__name__, path="/")

_DF = get_daily_kpi()

_LAYOUT = dbc.Container(
    [
        html.H3("Funnel Volume & Win‑rate"),
        dcc.DatePickerRange(
            id="date-filter",
            start_date=_DF["date"].min(),
            end_date=_DF["date"].max(),
            display_format="YYYY‑MM‑DD",
            minimum_nights=0,
            clearable=False,
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="chart-leads"), width=12),
                dbc.Col(dcc.Graph(id="chart-win"),   width=12),
                dbc.Col(dcc.Graph(id="chart-speed"), width=12),
            ],
            className="g-4",
        ),
    ],
    fluid=True,
)

layout = _LAYOUT  # Dash expects a `layout` var at module level


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------
@callback(
    Output("chart-leads", "figure"),
    Output("chart-win", "figure"),
    Output("chart-speed", "figure"),
    Input("date-filter", "start_date"),
    Input("date-filter", "end_date"),
)
def _update_timeseries(start: str, end: str):
    df = _DF.query("date >= @start and date <= @end")

    fig_leads = px.bar(
        df, x="date", y="leads",
        title="Leads per day",
        labels={"leads": "Count", "date": ""}
    )
    fig_leads.update_traces(marker_color=COLOR_PRIMARY)
    fig_win   = px.line(df, x="date", y="win_rate", title="7‑day rolling win‑rate")
    fig_win.update_yaxes(tickformat=".0%")
    fig_speed = px.line(
        df,
        x="date",
        y="median_mins_roll",
        title="7‑day rolling median minutes (Lead → 1st Action)",
    )
    return fig_leads, fig_win, fig_speed
