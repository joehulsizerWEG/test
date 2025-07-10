import datetime
from zoneinfo import ZoneInfo

import dash
from dash import Dash, dcc, html, Input, Output, callback_context
import plotly.express as px
import pandas as pd

# -------- Backend: your pipeline ---------------------------------
from tools.insured_mine.insured_mine_data import InsuredMindWorker,StatusType

# Path / filenames – parametrize or env‑vars as you like
FILE_DATE   = datetime.date(2025, 6, 25)          # default snapshot
CARDS_XLSX  = f"C:/users/Luiso/data/insured_mine/Dealcards_{FILE_DATE:%m_%d_%Y}.xlsx"
ACCTS_CSV   = f"C:/users/Luiso/data/insured_mine/Accounts_{FILE_DATE:%m_%d_%Y}.csv"
SEGMENTS_CSV=  "C:/users/Luiso/data/census/SegmentsAll_20180809.csv"

# Initialise *once* at server start‑up
worker = InsuredMindWorker(
    card_file_name=CARDS_XLSX,
    account_file_name=ACCTS_CSV,
    segment_file_name=SEGMENTS_CSV,
)
df_full = worker.data     # one unified dataframe; keep it read‑only

# Pre‑compute daily aggregates for speed --------------------------------------
def make_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy dataframe with the three core metrics by date."""
    # Daily lead count
    daily = (
        df.groupby("creation_date")
          .agg(
              leads=("TL_id", "size"),
              quotes=("Quoted Date", lambda s: s.notna().sum()),
              wins=("is_won", "sum"),
              median_mins=("CreationToFirstAction_min", "median")
          )
          .sort_index()
    )
    # Rolling win‑rate (quotes → wins)
    daily["win_rate"] = (
        daily["wins"].rolling(7, min_periods=1).sum()
        / daily["quotes"].rolling(7, min_periods=1).sum()
    )
    # Rolling median of speed
    daily["median_mins_roll"] = (
        daily["median_mins"].rolling(7, min_periods=1).median()
    )
    daily = daily.reset_index().rename(columns={"creation_date": "date"})
    return daily

daily_kpi = make_timeseries(df_full)

# -------- Front‑end ----------------------------------------------------------
app: Dash = dash.Dash(__name__, title="InsuredMine Dashboard")
server = app.server   # gunicorn entry‑point, if deploying

# --- Layout ------------------------------------------------------------------
app.layout = html.Div(
    [
        html.H1("Sales Process Overview – FloridaInsuranceQuotes.net", className="header"),
        dcc.DatePickerRange(
            id="date-range",
            start_date=daily_kpi["date"].min(),
            end_date=daily_kpi["date"].max(),
            minimum_nights=0,
            display_format="YYYY‑MM‑DD",
            clearable=False,
        ),
        html.Div(
            [
                dcc.Graph(id="ts-leads", className="fourwide"),
                dcc.Graph(id="ts-winrate", className="fourwide"),
                dcc.Graph(id="ts-speed", className="fourwide"),
            ],
            className="chart‑grid",
        ),
        html.Footer("© 2025 We Insure Analytics • Built with Plotly Dash", className="footer"),
    ]
)

# --- Callbacks ---------------------------------------------------------------
@app.callback(
    [Output("ts-leads", "figure"),
     Output("ts-winrate", "figure"),
     Output("ts-speed", "figure")],
    [Input("date-range", "start_date"),
     Input("date-range", "end_date")],
)
def update_charts(start_date, end_date):
    """Re‑draw all three charts when the user moves the date picker."""
    d0 = pd.to_datetime(start_date).date()
    d1 = pd.to_datetime(end_date).date()

    kpi = daily_kpi.query("date >= @d0 and date <= @d1")

    fig_leads = px.bar(
        kpi, x="date", y="leads",
        title="Leads created per day",
        labels={"leads": "Count", "date": ""}
    )

    fig_win = px.line(
        kpi, x="date", y="win_rate",
        title="7‑day rolling quote → close win rate",
        labels={"win_rate": "Win rate", "date": ""},
    )
    fig_win.update_yaxes(tickformat=".0%")

    fig_speed = px.line(
        kpi, x="date", y="median_mins_roll",
        title="7‑day rolling median minutes (Lead → 1st Action)",
        labels={"median_mins_roll": "Minutes", "date": ""},
    )

    # Uniform look & feel
    for fig in (fig_leads, fig_win, fig_speed):
        fig.update_layout(margin=dict(l=40, r=20, t=60, b=40))

    return fig_leads, fig_win, fig_speed

# --- Main --------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=8050)
