# app.py
"""
Streamlit front‑end for the InsuredMindWorker analytics toolkit.

Usage:
  streamlit run app.py
"""

from pathlib import Path
from datetime import datetime, date
from zoneinfo import ZoneInfo
import tempfile, uuid

import pandas as pd
import streamlit as st

# -------------------------------
#  Load your analytics engine
# -------------------------------
# Put your original file in the same folder, minus the __main__ guard.
from tools.insured_mine.insured_mine_data import InsuredMindWorker,StatusType, EASTERN
# -------------------------------------------------
#  1.  Configuration & global helpers
# -------------------------------------------------
FILE_DATE_FMT = "%m_%d_%Y"

@st.cache_data(show_spinner=False, ttl="12h")
def load_worker(card_xlsx: Path,
                account_csv: Path,
                segment_csv: Path) -> InsuredMindWorker:
    """Initialise InsuredMindWorker once & cache the heavy data wrangling."""
    return InsuredMindWorker(
        card_file_name=str(card_xlsx),
        account_file_name=str(account_csv),
        segment_file_name=str(segment_csv)
    )

@st.cache_data(show_spinner=False)
def build_daily_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate counts for core time‑series."""
    grp = df.groupby("creation_date").agg(
        leads=("TL_id", "nunique"),
        won=("is_won", "sum"),
        quoted=("is_quoted", "sum"),
        contacted=("is_contacted", "sum")
    )
    grp["win_rate"] = grp["won"] / grp["leads"]
    return grp.sort_index()

# -------------------------------------------------
#  2.  Sidebar – data upload & filters
# -------------------------------------------------
st.sidebar.title("📊 Data files")

def file_uploader(label, key, type_):
    return st.sidebar.file_uploader(label, type=type_, key=key)

card_file   = file_uploader("Deal Cards (Excel)", "cards", ["xlsx"])
account_file = file_uploader("Accounts (CSV)", "accts", ["csv"])
segment_file = file_uploader("Segment map (CSV)", "segs", ["csv"])

if not (card_file and account_file and segment_file):
    st.sidebar.info("Upload **all three files** to begin.")
    st.stop()

def _get_session_id() -> str:
    """
    Return the current Streamlit *session_id* if available,
    otherwise fall back to a random UUID.
    Works on Streamlit ≥1.27 and older versions.
    """
    try:
        # Newer releases
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        if ctx is not None:
            return ctx.session_id
    except ModuleNotFoundError:
        pass                          # pre‑1.27 handled below

    # Legacy API (≤1.26)
    try:
        import streamlit as st
        return st.experimental_get_script_run_ctx().session_id   # type: ignore[attr-defined]
    except Exception:
        # Fallback – e.g. when running via `streamlit run` with no active ctx
        return uuid.uuid4().hex


# Convert uploads to temp files so InsuredMindWorker can open via path
tmp_dir = Path(tempfile.gettempdir()) / f"ime_{_get_session_id()}"
tmp_dir.mkdir(exist_ok=True)

card_path    = tmp_dir / "cards.xlsx";   card_path.write_bytes(card_file.getvalue())
account_path = tmp_dir / "accts.csv";    account_path.write_bytes(account_file.getvalue())
segment_path = tmp_dir / "segs.csv";     segment_path.write_bytes(segment_file.getvalue())

# Optional high‑level filters
st.sidebar.divider()
st.sidebar.subheader("Filters")

with st.sidebar:
    col1, col2 = st.columns(2)
    start = col1.date_input("Start date", value=date(2025, 1, 1), key="start")
    end   = col2.date_input("End date", value=date(2025, 12, 31), key="end")
    show_bh = st.checkbox("Business hours only", value=False)

# -------------------------------------------------
#  3.  Main – landing page
# -------------------------------------------------
st.title("🧮 InsuredMine Overview")

with st.spinner("Crunching the data …"):
    ime = load_worker(card_path, account_path, segment_path)
    df  = ime._standard_filter(
            ime.data,
            status_type=StatusType.ALL,
            start_date=datetime.combine(start, datetime.min.time(), tzinfo=EASTERN),
            end_date=datetime.combine(end, datetime.max.time(), tzinfo=EASTERN),
            is_business_hours=show_bh
          )
    ts = build_daily_counts(df)

# ---- KPI cards ----------------------------------------------------
kpi_cols = st.columns(4)
kpi_cols[0].metric("Total Leads",      f"{ts['leads'].sum():,}")
kpi_cols[1].metric("Total Won",        f"{ts['won'].sum():,}")
kpi_cols[2].metric("Overall Win %",    f"{ts['won'].sum()/ts['leads'].sum():.1%}")
kpi_cols[3].metric("Median Δ to 1st Action (min)",
                   f"{df['CreationToFirstAction_min'].median():.1f}")

st.divider()

# ---- Time‑series visualisations -----------------------------------
st.subheader("Daily Lead Volume")
st.line_chart(ts["leads"])

st.subheader("Daily Win Rate")
st.line_chart(ts["win_rate"])

st.subheader("Median Minutes to First Action")
first_action_daily = df.groupby("creation_date")["CreationToFirstAction_min"].median()
st.line_chart(first_action_daily)

# ---- Data preview -------------------------------------------------
with st.expander("Raw data preview"):
    st.dataframe(df.head(200))

# -------------------------------------------------
#  4.  Navigation / further analysis (optional)
# -------------------------------------------------
st.markdown("""
### Next steps

* Use the **☰ sidebar** to drill in further:
  * Toggle business‑hours view
  * Narrow the date window  
* Build new pages (e.g. *Agent Performance*, *Source Attribution*) by
  calling the rich `get_*` helpers already shipped in `InsuredMindWorker`.
""")
