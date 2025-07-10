# app.py -----------------------------------------------------------
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from datetime import date
from zoneinfo import ZoneInfo
from typing import Optional, List
import plotly.graph_objects as go
from mpl_toolkits.mplot3d.art3d import collection_2d_to_3d
from plotly.subplots import make_subplots
import tempfile, uuid
import numpy as np
from streamlit import multiselect

# Weekday order helper
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"]


# ------------------------------------------------------------------
# 1.  Load data once per session
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading CRM data …")
def load_data() -> pd.DataFrame:
    file_date  = date(2025, 7, 7)
    cards_xlsx = Path(f"C:/users/Luiso/data/insured_mine/Dealcards_{file_date:%m_%d_%Y}.xlsx")
    accounts   = Path(f"C:/users/Luiso/data/insured_mine/Accounts_{file_date:%m_%d_%Y}.csv")
    segments   = Path("C:/users/Luiso/data/census/SegmentsAll_20180809.csv")

    from toolshed.tools.insured_mine.insured_mine_data import InsuredMindWorker
    ime = InsuredMindWorker(cards_xlsx, accounts, segments)
    return ime.data      # quote‑level dataframe

df_full = load_data()

# ------------------------------------------------------------------
# 2.  Sidebar – global filters
# ------------------------------------------------------------------
st.sidebar.header("Global Filters")

# ---- ①  Which date field drives the range filter? ----------------
date_field = st.sidebar.radio(
    "Filter by …",
    options=("Creation Date", "Closed Date"),
    index=0                           # 0 ⇒ default to Creation Date
)
date_col = "Creation Date" if date_field == "Creation Date" else "Closed Date"

# ---- ②  The rest of your global filters --------------------------
biz_hours_only = st.sidebar.checkbox("Business hours only", value=False)

state_sel   = st.sidebar.multiselect(
    "State", sorted(df_full["State"].dropna().unique())
)
source_category_sel = st.sidebar.multiselect(
    "Source Category", sorted(df_full["Source Category"].dropna().unique())
)
source_sel  = st.sidebar.multiselect(
    "Source", sorted(df_full["Source Name"].dropna().unique())
)
agent_sel   = st.sidebar.multiselect(
    "Primary Agent", sorted(df_full["PrimaryAgent"].dropna().unique())
)
product_category_sel = st.sidebar.multiselect(
    "Product Category", sorted(df_full["Product"].dropna().unique())
)
product_sel = st.sidebar.multiselect(
    "Category", sorted(df_full["Category"].dropna().unique())
)
status_sel = st.sidebar.multiselect(
    "Status", sorted(df_full["Status"].dropna().unique())
)

# ---- ③  Build the dynamic date‑range widget ----------------------
#     • It always shows the min/max for the *selected* column.
#     • .dt.date ensures the widget gets date objects, not datetimes.
df_full[date_col] = pd.to_datetime(df_full[date_col], errors="coerce")  # just once is safe
min_date = df_full[date_col].min().date()
max_date = df_full[date_col].max().date()

start_d, end_d = st.sidebar.date_input(
    f"{date_field} range",            # label changes with the toggle
    (min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# ---- ④  Helper to apply all filters ------------------------------
def apply_filters(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    # start with the date mask; ignore rows where the chosen date is NaT
    msk = (
        df[date_col].notna() &
        df[date_col].dt.date.between(start_d, end_d)
    )
    if biz_hours_only:
        msk &= df["creation_is_business_open"]  # adjust if you track biz hours for "closed"
    if state_sel:
        msk &= df["State"].isin(state_sel)
    if source_category_sel:
        msk &= df["Source Category"].isin(source_category_sel)
    if source_sel:
        msk &= df["Source Name"].isin(source_sel)
    if agent_sel:
        msk &= df["PrimaryAgent"].isin(agent_sel)
    if product_category_sel:
        msk &= df["Product"].isin(product_category_sel)
    if product_sel:
        msk &= df["Category"].isin(product_sel)
    if status_sel:
        msk &= df["Status"].isin(status_sel)
    return df.loc[msk].copy()

# ---- ⑤  Get the filtered frame -----------------------------------
df = apply_filters(df_full, date_col)

# universal table‑printing helper --------------------------------------------
def show_table(
    df: pd.DataFrame,
    pct_cols: Optional[List[str]] = None,
    number_cols : Optional[List[str]] = None,
    currency_cols: Optional[List[str]] = None,
    currency_fmt: str = "${:,.2f}",              # configurable pattern
):
    """
    Display a dataframe with formatted percentage and currency columns.

    - pct_cols      : list of columns that contain proportions 0‑1 → shown as %.
                      If None, infer columns whose values are all between 0‑1.
    - currency_cols : list of numeric columns that should render as dollars.
                      If None, leave them unformatted.
    """
    if pct_cols is None:
        pct_cols = [c for c in df.columns
                    if pd.api.types.is_numeric_dtype(df[c])
                    and df[c].between(0, 1).all()]

    fmt = {c: '{:.1%}' for c in pct_cols}

    if number_cols:
        fmt.update({c: '{:,}' for c in number_cols})

    if currency_cols is not None:
        fmt.update({c: currency_fmt for c in currency_cols})

    st.dataframe(df.style.format(fmt))

def _contact_to_close(g):
    """% of contacted leads that became wins for ONE agent group `g`."""
    if g['is_contacted'].sum() == 0:
        return np.nan                 # avoids divide‑by‑zero
    return g.loc[g['is_contacted'], 'is_won'].mean()

def _quote_to_close(g):
    """% of quoted leads that became wins for ONE agent group `g`."""
    if g['is_quoted'].sum() == 0:
        return np.nan
    return g.loc[g['is_quoted'], 'is_won'].mean()


def _product_mix(df):
    """Return a wide table of Estimated Premium $ per agent × product."""
    df = df.loc[df['is_won']]
    wide = (
        df.pivot_table(index="PrimaryAgent",
                       columns="Product",
                       values="Expected Premium",
                       aggfunc="sum",
                       fill_value=0)
          .assign(Total=lambda d: d.sum(axis=1))        # row totals
          .sort_values("Total", ascending=False)
          .reset_index()
    )

    # Grand total row like the yellow one in the screenshot
    grand_total = wide.drop(columns="PrimaryAgent").sum().to_frame().T
    grand_total.insert(0, "PrimaryAgent", "Total Premium")
    wide = pd.concat([wide, grand_total], ignore_index=True)

    return wide

# ------------------------------------------------------------------
# 3.  Navigation
# ------------------------------------------------------------------
PAGES = {
    "Executive Summary": "page_exec",
    "Pipeline":          "page_pipeline",
    "Agents":            "page_agents",
    "Lead Sources":      "page_sources",
    "Speed‑to‑Lead":     "page_speed",
    "Products":          "page_products",
    "Temporal":          "page_temporal",
    "Geography":         "page_geo",
    "Call Center":       "page_calls",
    "Data Quality":      "page_quality",

}

page = st.sidebar.radio("Go to page", list(PAGES.keys()))
st.title(page)

# ------------------------------------------------------------------
# 4.  Page definitions
# ------------------------------------------------------------------
def page_exec(df: pd.DataFrame) -> None:
    """Executive summary dashboard with Lead/Quote toggle, Day/Week/Month grains,
    stacked Close‑Date chart, and enhanced line charts."""

    # ─────────────────────  0. SIDEBAR OPTIONS  ─────────────────────
    metric_basis = st.sidebar.radio(
        "Show metrics by …", ["Leads", "Quotes"], index=0
    )
    grp_unit = st.sidebar.radio("Group timeline by", ["Day", "Week", "Month"], index=0)

    df['won_premium'] = df['Expected Premium']*df['is_won']
    # ─────────────────────  1. NORMALISE TO ONE ROW PER ENTITY  ─────────────────────
    if metric_basis == "Leads":
        working_df = (
            df.groupby("lead_id", as_index=False)
              .agg(
                  Creation_Date=("Creation Date", "min"),
                  is_contacted=("is_contacted", "max"),
                  is_quoted   =("is_quoted",    "max"),
                  is_won      =("is_won",       "max"),
                  won_premium =("won_premium",  "sum")
              )
              .rename(columns={"lead_id": "entity_id"})
        )
    else:
        working_df = (
            df.drop_duplicates("quote_id")
              .loc[:, ["quote_id", "Creation Date",
                       "is_contacted", "is_quoted", "is_won","won_premium"]]
              .rename(columns={"quote_id": "entity_id",
                               "Creation Date": "Creation_Date"})
        )

    # ─────────────────────  2. TOP‑LINE METRICS  ─────────────────────
    total_entities = len(working_df)
    contacted = working_df["is_contacted"].sum()
    quoted    = working_df["is_quoted"].sum()
    won       = working_df["is_won"].sum()
    premium_won = working_df["won_premium"].sum()
    premium_k = premium_won / 1_000  # convert to thousands

    col1, col2, col3,col4 = st.columns(4)
    col1.metric("Won", f"{won:,}")
    col2.metric("Quoted",  f"{quoted:,}")
    col3.metric(metric_basis, f"{total_entities:,}")

    col4.metric("Premium ($K)", f"${premium_k:,.1f}K")


    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Win %", f"{won / total_entities:0.1%}")
    col2.metric("Quote %", f"{quoted / total_entities:0.1%}")
    col3.metric("Contact %",         f"{contacted / total_entities:0.1%}")
    col4.metric("QuoteToClose %", f"{won / quoted:0.1%}")


    # ─────────────────────  3. TIME‑KEY FOR DAY/WEEK/MONTH  ─────────────────────
    if grp_unit == "Day":
        working_df["time_key"] = working_df["Creation_Date"].dt.date
        x_name = "date"
    elif grp_unit == "Week":
        # Monday as start of week
        working_df["time_key"] = (
            working_df["Creation_Date"].dt.to_period("W-MON").dt.start_time
        )
        x_name = "week"
    else:  # Month
        working_df["time_key"] = (
            working_df["Creation_Date"].dt.to_period("M").dt.to_timestamp()
        )
        x_name = "month"

    # ─────────────────────  4. DAILY / WEEKLY / MONTHLY AGGREGATION  ─────────────────────
    daily = (
        working_df
        .groupby("time_key", as_index=False)
        .agg(
            entities  =("entity_id",    "nunique"),
            contacted =("is_contacted", "sum"),
            quoted    =("is_quoted",    "sum"),
            wins      =("is_won",       "sum")
        )
        .rename(columns={"time_key": x_name})
    )

    daily["contact_rate"]            = daily["contacted"] / daily["entities"]
    daily["quote_rate"]              = daily["quoted"]    / daily["entities"]
    daily["close_rate"]              = daily["wins"]      / daily["entities"]
    daily["quote_rate_from_contact"] = (
        daily["quoted"] / daily["contacted"].replace(0, pd.NA)
    )
    daily["close_rate_from_quote"]   = (
        daily["wins"]   / daily["quoted"].replace(0, pd.NA)
    )
    daily["close_rate_from_contact"] = (
        daily["wins"]   / daily["contacted"].replace(0, pd.NA)
    )

    # ─────────────────────  5‑D  OPTIONAL LEAD‑STATUS STACKED BAR (CREATION)  ─────────────────────
    if "Status" in df.columns and metric_basis == "Leads":
        if grp_unit == "Day":
            df["time_key"] = df["Creation Date"].dt.date
        elif grp_unit == "Week":
            df["time_key"] = df["Creation Date"].dt.to_period("W-MON").dt.start_time
        else:
            df["time_key"] = df["Creation Date"].dt.to_period("M").dt.to_timestamp()

        status_grp2 = (
            df.groupby(["time_key", "Status"]).size()
              .reset_index(name="leads")
              .rename(columns={"time_key": x_name})
        )
        fig_status = px.bar(
            status_grp2, x=x_name, y="leads", color="Status",
            title="Leads by Status (Creation)", labels={"leads": "# Leads"},
            barmode="stack"
        )
        st.plotly_chart(fig_status, use_container_width=True)

    # ─────────────────────  5‑C  CLOSED‑DATE STACKED BAR (Status)  ─────────────────────
    if {"Closed Date", "Status"}.issubset(df.columns):
        closed_df = df.dropna(subset=["Closed Date"]).copy()

        # Pick the same grain (Day / Week / Month) the user selected for the rest of the dashboard
        if grp_unit == "Day":
            closed_df["close_key"] = closed_df["Closed Date"].dt.date
        elif grp_unit == "Week":
            closed_df["close_key"] = (
                closed_df["Closed Date"].dt.to_period("W-MON").dt.start_time
            )
        else:  # Month
            closed_df["close_key"] = (
                closed_df["Closed Date"].dt.to_period("M").dt.to_timestamp()
            )

        status_grp = (
            closed_df
            .groupby(["close_key", "Status"])
            .size()
            .reset_index(name="count")
            .sort_values("close_key")
            .rename(columns={"close_key": x_name})
        )

        fig_close_bar = px.bar(
            status_grp,
            x=x_name, y="count", color="Status",
            barmode="stack",
            title=f"Closed Records by Status ({grp_unit})",
            labels={"count": "# Closed"}
        )
        st.plotly_chart(fig_close_bar, use_container_width=True)


    else:
        st.warning("No 'Closed Date' and/or 'Status' columns — closed‑date charts skipped.")

    # ─────────────────────  5‑C  CLOSED‑DATE STACKED BAR (Status)  ─────────────────────
    if {"Closed Date", "Product"}.issubset(df.columns):
        closed_df = df.dropna(subset=["Closed Date"]).copy()
        closed_df = closed_df.loc[closed_df['is_won']]

        # Pick the same grain (Day / Week / Month) the user selected for the rest of the dashboard
        if grp_unit == "Day":
            closed_df["close_key"] = closed_df["Closed Date"].dt.date
        elif grp_unit == "Week":
            closed_df["close_key"] = (
                closed_df["Closed Date"].dt.to_period("W-MON").dt.start_time
            )
        else:  # Month
            closed_df["close_key"] = (
                closed_df["Closed Date"].dt.to_period("M").dt.to_timestamp()
            )

        status_grp = (
            closed_df
            .groupby(["close_key", "Product"])
            .size()
            .reset_index(name="count")
            .sort_values("close_key")
            .rename(columns={"close_key": x_name})
        )

        fig_close_bar = px.bar(
            status_grp,
            x=x_name, y="count", color="Product",
            barmode="stack",
            title=f"Won by Product ({grp_unit})",
            labels={"count": "# Closed"}
        )
        st.plotly_chart(fig_close_bar, use_container_width=True)


    else:
        st.warning("No 'Closed Date' and/or 'Status' columns — closed‑date charts skipped.")

    # ─────────────────────  5‑D  OPTIONAL LEAD‑STATUS STACKED BAR (CREATION)  ─────────────────────
    if "Status" in df.columns and metric_basis == "Leads":
        if grp_unit == "Day":
            df["time_key"] = df["Creation Date"].dt.date
        elif grp_unit == "Week":
            df["time_key"] = df["Creation Date"].dt.to_period("W-MON").dt.start_time
        else:
            df["time_key"] = df["Creation Date"].dt.to_period("M").dt.to_timestamp()

        status_grp2 = (
            df.groupby(["time_key", "Status"]).size()
              .reset_index(name="leads")
              .rename(columns={"time_key": x_name})
        )
        fig_status = px.bar(
            status_grp2, x=x_name, y="leads", color="Status",
            title="Leads by Status (Creation)", labels={"leads": "# Leads"},
            barmode="stack"
        )
        st.plotly_chart(fig_status, use_container_width=True,key='status_chart')

    # ─────────────────────  5‑A  OPTIONAL LEAD‑SOURCE STACKED BAR (CREATION)  ─────────────────────
    if "Source Category" in df.columns and metric_basis == "Leads":
        if grp_unit == "Day":
            df["time_key"] = df["Creation Date"].dt.date
        elif grp_unit == "Week":
            df["time_key"] = df["Creation Date"].dt.to_period("W-MON").dt.start_time
        else:
            df["time_key"] = df["Creation Date"].dt.to_period("M").dt.to_timestamp()

        source_grp2 = (
            df.groupby(["time_key", "Source Category"]).size()
              .reset_index(name="leads")
              .rename(columns={"time_key": x_name})
        )
        fig_source = px.bar(
            source_grp2, x=x_name, y="leads", color="Source Category",
            title="Leads by Source (Creation)", labels={"leads": "# Leads"},
            barmode="stack"
        )
        st.plotly_chart(fig_source, use_container_width=True,key='source_chart')

    # ─────────────────────  5‑A  CONTACT‑, QUOTE‑ & WIN‑RATE COMBINED  ─────────────────────
    totals_long = daily.melt(
        id_vars=[x_name],
        value_vars=["contact_rate", "quote_rate", "close_rate"],
        var_name="Metric", value_name="value"
    )
    totals_long["Metric"] = totals_long["Metric"].map({
        "contact_rate": "Contact Rate",
        "quote_rate":   "Quote Rate",
        "close_rate":   "Win % (Close Rate)"
    })

    fig_totals = px.line(
        totals_long, x=x_name, y="value", color="Metric",
        title=f"Contact Rate, Quote Rate & Win % of Total {metric_basis}",
        markers=True, labels={"value": "%"}
    )
    fig_totals.update_yaxes(tickformat=".0%")
    fig_totals.update_traces(mode="lines+markers")
    st.plotly_chart(fig_totals, use_container_width=True)

    # ─────────────────────  5‑B  CONTACT‑>QUOTE & CONTACT‑>WIN COMBINED  ─────────────────────
    contact_long = daily.melt(
        id_vars=[x_name],
        value_vars=["quote_rate_from_contact", "close_rate_from_contact"],
        var_name="Metric", value_name="value"
    )
    contact_long["Metric"] = contact_long["Metric"].map({
        "quote_rate_from_contact": "Quote Rate (Contacted → Quoted)",
        "close_rate_from_contact": "Win % (Contacted → Won)"
    })

    fig_contact = px.line(
        contact_long, x=x_name, y="value", color="Metric",
        title="Quote & Win % for Contacted Records",
        markers=True, labels={"value": "%"}
    )
    fig_contact.update_yaxes(tickformat=".0%")
    fig_contact.update_traces(mode="lines+markers")
    st.plotly_chart(fig_contact, use_container_width=True)





def page_pipeline(df: pd.DataFrame) -> None:
    """Conversion‑funnel & time‑metrics dashboard."""

    # ------------------------------------------------------------------
    # 1. CONVERSION FUNNEL + TABLE (single table, right‑justified, %, win‑rate)
    # ------------------------------------------------------------------
    st.subheader("Conversion Funnel")
    stages = [
        "New Lead", "Attempted Contact", "Gathering Info",
        "Quoting", "Quote Sent", "Policy Binding",
        "Policy Issued", "is_won"
    ]

    # counts of leads that *reached* each stage
    counts = [df[f"Moved On {s}"].notna().sum() for s in stages[:-1]]
    counts.append(df["is_won"].sum())

    # % of original leads that reach each stage
    pct_of_total = np.divide(counts, counts[0], where=np.array(counts) > 0)

    # conditional win‑rate among leads that made it to the stage
    wins = counts[-1]
    conditional_win = [
        df.loc[df[f"Moved On {s}"].notna(), "is_won"].sum() / c if c else 0
        for s, c in zip(stages[:-1], counts[:-1])
    ] + [1.0]          # for the final “is_won” row

    # master table (includes win‑rate)
    funnel_df = pd.DataFrame(
        {
            "Stage": stages,
            "% of Leads": pct_of_total,
            "Win‑rate | reached stage": conditional_win
        }
    )

    # ------------ CHART  (no win‑rate column passed) ------------
    chart_df = funnel_df.drop(columns=["Win‑rate | reached stage"])

    fig = px.funnel(
        chart_df,
        x="% of Leads",
        y="Stage",
        text=chart_df["% of Leads"].apply(lambda x: f"{x:.2%}")
    )
    fig.update_traces(textposition="inside",texttemplate="%{text}")
    fig.update_xaxes(tickformat=".1%")
    st.plotly_chart(fig, use_container_width=True)

    # ------------ TABLE  (still shows win‑rate) ------------------
    st.dataframe(
        funnel_df.style
        .format({"% of Leads": "{:.2%}",
                 "Win‑rate | reached stage": "{:.2%}"})
        .set_properties(**{"text-align": "right"})
        .set_table_styles([{"selector": "th",
                            "props": [("text-align", "right")]}]),
        use_container_width=True
    )

    # ------------------------------------------------------------------
    # 2. TIME‑IN‑STAGE DISTRIBUTION (slider, 5‑95 default, box w/o whiskers)
    # ------------------------------------------------------------------
    st.subheader("Time in Stage")

    # --- percentile range selector ---
    lo, hi = st.slider(
        "Percentile filter (inclusive)",
        min_value=0, max_value=100, value=(5, 95), step=1
    )

    # --- NEW: ensure we have a win‑only column -------------------------
    win_col = "CreationToClosed_min"  # raw column name
    #   keep times only for won deals; others become NaN so they’re ignored
    df[win_col] = df[win_col].where(df["is_won"], np.nan)

    # --- list of all timing columns, now including the win metric -------
    box_cols = [
                   f"CreationTo{stage.replace(' ', '')}_min" for stage in
                   ["NewLead", "AttemptedContact", "GatheringInfo",
                    "Quoting", "QuoteSent", "PolicyBinding", "PolicyIssued"]
               ] + [win_col]  # ← append in desired order (last row/box will be “win”)

    melted = df.melt(value_vars=box_cols, var_name="Stage", value_name="Minutes")

    # ---- make Stage a *categorical* with the desired order ------------
    melted["Stage"] = pd.Categorical(melted["Stage"],
                                     categories=box_cols, ordered=True)

    # ---- percentile filter (unchanged) --------------------------------
    def filter_by_percentile(group: pd.DataFrame) -> pd.DataFrame:
        group = group.dropna(subset=["Minutes"])
        low_val, high_val = np.percentile(group["Minutes"], [lo, hi])
        return group[(group["Minutes"] >= low_val) & (group["Minutes"] <= high_val)]

    filt = melted.groupby("Stage", group_keys=False).apply(filter_by_percentile)

    # ------------------------------------------------------------------
    # SUMMARY TABLE (respect order)
    # ------------------------------------------------------------------
    summary = (
        filt.groupby("Stage")["Minutes"]
            .agg(P25=lambda s: s.quantile(0.25),
                 Median="median",
                 P75=lambda s: s.quantile(0.75))
            .reset_index()
            .sort_values("Stage")              # keeps categorical order
    )

    # format + right‑align
    st.dataframe(
        summary.style
               .format({"P25": "{:,.2f}", "Median": "{:,.2f}", "P75": "{:,.2f}"})
               .set_properties(**{"text-align": "right"})
               .set_table_styles([{"selector": "th",
                                   "props": [("text-align", "right")]}]),
        use_container_width=True
    )

    # ------------------------------------------------------------------
    # BOX‑PLOT  (draw traces *in box_cols order*)
    # ------------------------------------------------------------------
    fig2 = go.Figure()

    for stage in box_cols:                     # iterate in the desired order
        grp = filt[filt["Stage"] == stage]
        if grp.empty:
            continue
        q1  = grp["Minutes"].quantile(0.25)
        med = grp["Minutes"].median()
        q3  = grp["Minutes"].quantile(0.75)

        fig2.add_trace(
            go.Box(
                q1=[q1], median=[med], q3=[q3],
                lowerfence=[q1], upperfence=[q3],
                x=[stage],                     # anchor on its own category
                name=stage,
                orientation="v",
                boxpoints=False
            )
        )

    fig2.update_layout(
        boxmode="group",
        yaxis_title="Minutes",
        xaxis_title="",
        showlegend=False,
        xaxis=dict(categoryorder="array", categoryarray=box_cols)  # enforce order on axis
    )

    st.plotly_chart(fig2, use_container_width=True)
    # ------------------------------------------------------------------
    # 3. MEDIAN TIME‑TO‑STAGE BY PRODUCT
    # ------------------------------------------------------------------
    st.subheader("Median Time‑to‑Stage by Product")

    # --- remelt to retain Product --------------------------------------
    melt_prod = df.melt(
        id_vars=["Product"],  # keep Product column
        value_vars=box_cols,  # CreationTo… columns
        var_name="Stage",
        value_name="Minutes"
    )

    # ensure Stage uses the ordered categorical from box_cols
    melt_prod["Stage"] = pd.Categorical(melt_prod["Stage"],
                                        categories=box_cols,
                                        ordered=True)

    # --- apply the same percentile filter, stage‑wise ------------------
    def pct_filter(group: pd.DataFrame) -> pd.DataFrame:
        group = group.dropna(subset=["Minutes"])
        if group.empty:
            return group  # nothing to filter
        low, high = np.percentile(group["Minutes"], [lo, hi])
        return group[(group["Minutes"] >= low) & (group["Minutes"] <= high)]

    filt_prod = melt_prod.groupby("Stage", group_keys=False).apply(pct_filter)

    # --- build pivot: rows = Stage, cols = Product, values = median ----
    pivot = (
        filt_prod.groupby(["Stage", "Product"])["Minutes"]
        .median()
        .unstack("Product")  # Products become columns
        .reindex(box_cols)  # enforce row order
    )

    # --- display with styling ------------------------------------------
    st.dataframe(
        pivot.style
        .format("{:,.2f}")  # 0,000.00
        .set_properties(**{"text-align": "right"})
        .set_table_styles([{"selector": "th",
                            "props": [("text-align", "right")]}]),
        use_container_width=True
    )


def page_agents(df):
    st.subheader("Agent leaderboard")

    tbl = (
        df.assign(premium_won = df["Expected Premium"] * df["is_won"])  # NEW
          .groupby("PrimaryAgent")
          .apply(lambda g: pd.Series({
              "Wins": g["is_won"].sum(),
              "WinPct": g["is_won"].mean(),
              "contact_to_close": _contact_to_close(g),
              "quote_to_close": _quote_to_close(g),
              "Expected Premium": g["premium_won"].sum(),
              "Leads"            : len(g),
              "Contacts"         : g["is_contacted"].sum(),
              "ContactPct"       : g["is_contacted"].mean(),
              "Quoted"           : g["is_quoted"].sum(),
              "QuotedPct"        : g["is_quoted"].mean(),

          }))
          .sort_values("Wins", ascending=False)
          .reset_index()
    )

    pct_cols = [
        "WinPct", "ContactPct", "QuotedPct",
        "contact_to_close", "quote_to_close"
    ]

    number_cols = ['Leads','Contacts','Quoted','Wins']

    show_table(tbl, pct_cols=pct_cols,number_cols=number_cols,currency_cols=['Expected Premium'])

    st.subheader("Premium by product line")
    prod_tbl = _product_mix(df)
    show_table(prod_tbl, currency_cols=["Total"] + prod_tbl.columns.drop(["PrimaryAgent", "Total"]).tolist())


def page_sources(df):
    st.subheader("Lead sources – volume & funnel rates")

    tbl = (
        df.groupby("Source Category")
          .agg(Leads=("TL_id", "size"),
               Wins=("is_won", "sum"),
               ContactPct=("is_contacted", "mean"),
               QuotePct=("is_quoted", "mean"),
               WinPct=("is_won", "mean")
               )
          .reset_index()
          .sort_values("Leads", ascending=False)
    )
    show_table(tbl, pct_cols=["ContactPct", "QuotePct", "WinPct"])

    tbl = (
        df.groupby("Source Name")
          .agg(Leads=("TL_id", "size"),
               Wins=("is_won", "sum"),
               ContactPct=("is_contacted", "mean"),
               QuotePct=("is_quoted", "mean"),
               WinPct=("is_won", "mean")
               )
          .reset_index()
          .sort_values("Leads", ascending=False)
    )

    fig = px.bar(
        tbl, x="Source Name", y="Leads",
        hover_data=["ContactPct", "QuotePct", "WinPct"]
    )
    st.plotly_chart(fig, use_container_width=True)
    show_table(tbl, pct_cols=["ContactPct", "QuotePct", "WinPct"])



def page_speed(df: pd.DataFrame) -> None:
    st.subheader("Speed‑to‑Lead vs Win % (binned)")

    # ── UI CONTROLS ──────────────────────────────────────────────────────────────
    # 1️⃣  Outlier removal: percentile cap (default 95th)
    pct_cap = st.slider(
        "Exclude leads above this percentile of minutes‑to‑first‑action:",
        min_value=50, max_value=100, value=95, step=1,
        help="Rows whose 'CreationToFirstAction_min' value exceeds this percentile "
             "are dropped from the analysis."
    )

    # 2️⃣  Bucket count: choose number of quantile buckets
    bucket_options = [5, 10, 15, 20, 25, 30, 35, 40]
    q_buckets = st.selectbox(
        "Number of buckets (quantile bins):",
        options=bucket_options,
        index=bucket_options.index(10)  # keeps old behaviour as the default
    )

    # ── DATA PREP ────────────────────────────────────────────────────────────────
    df = df[df["CreationToFirstAction_min"].notna()]

    # Outlier filter
    max_allowed = df["CreationToFirstAction_min"].quantile(pct_cap / 100)
    df = df[df["CreationToFirstAction_min"] <= max_allowed]

    # Quantile binning
    df["bin"] = pd.qcut(df["CreationToFirstAction_min"],
                        q=q_buckets,
                        duplicates="drop")

    # ── AGGREGATION ──────────────────────────────────────────────────────────────
    agg = (
        df.groupby("bin")
          .agg(
              win=("is_won", "mean"),
              avg_min=("CreationToFirstAction_min", "mean"),
              leads=("TL_id", "size")
          )
          .reset_index()
    )

    # Extract bin edges for tooltip clarity
    agg["bin_min"] = agg["bin"].apply(lambda x: round(x.left, 3))
    agg["bin_max"] = agg["bin"].apply(lambda x: round(x.right, 3))
    agg = agg.drop(columns="bin").sort_values("avg_min")  # sort for a proper line

    # ── VISUALISATION ────────────────────────────────────────────────────────────
    fig = px.line(
        agg,
        x="avg_min",
        y="win",
        markers=True,              # show points on the line
        labels={
            "avg_min": "Avg minutes to first action",
            "win": "Win %"
        },
        hover_data=["leads", "bin_min", "bin_max"]
    )
    fig.update_yaxes(tickformat=".1%")

    st.plotly_chart(fig, use_container_width=True)

    # ── OPTIONAL: show underlying table ─────────────────────────────────────────
    show_table(agg, pct_cols=["win"])



def page_products(df):
    st.subheader("Product mix – treemap")
    df = df[df["Product"].notna() & (df["Product"] != "")]
    fig = px.treemap(df, path=["Product"], values="TL_id")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Product Category performance table")
    tbl = (
        df.groupby("Product")
          .agg(Leads=("TL_id","size"),
               Wins=("is_won", "sum"),
               WinPct=("is_won", "mean"),
               QuotePct=("is_quoted", "mean"),
               ContactPct=("is_contacted", "mean")
               )
          .reset_index()
          .sort_values("Leads", ascending=False)
    )
    show_table(tbl, pct_cols=["WinPct","QuotePct","ContactPct"])

    st.subheader("Product performance table")
    tbl = (
        df.groupby("Category")
          .agg(Leads=("TL_id","size"),
               Wins=("is_won", "sum"),
               WinPct=("is_won", "mean"),
               QuotePct=("is_quoted", "mean"),
               ContactPct=("is_contacted", "mean")
               )
          .reset_index()
          .sort_values("Leads", ascending=False)
    )
    show_table(tbl, pct_cols=["WinPct","QuotePct","ContactPct"])

def page_temporal(df):
    st.subheader("Performance by Day & Hour")

    # ------------------------------------------------------------------
    # 1.  Make sure the remaining hours are padded out with every integer
    #     hour between the smallest and largest hour that survived
    #     the upstream filters (e.g. Is_business_hour).
    # ------------------------------------------------------------------
    hours_int = df["creation_hour"].astype(int)          # e.g.  8, 9, 10 …
    if hours_int.empty:
        # Fallback in the very rare case the frame is empty
        hour_order = [f"{h:02d}" for h in range(24)]
    else:
        start_hr, end_hr = hours_int.min(), hours_int.max()
        hour_order = [f"{h:02d}" for h in range(start_hr, end_hr + 1)]

    # ------------------------------------------------------------------
    # 2.  Day ordering stays the same
    # ------------------------------------------------------------------
    df["creation_day_name"] = pd.Categorical(
        df["creation_day_name"], categories=WEEKDAYS, ordered=True
    )
    df["creation_hour"] = df["creation_hour"].astype(int).astype(str).str.zfill(2)

    # ------------------------------------------------------------------
    # 3.  Helper for the annotated heat‑maps
    # ------------------------------------------------------------------
    def make_heatmap(data, value_col, label_fmt, colorscale, title,
                     zformat=".0%", agg_fn=None):

        # Default aggregation if caller does not specify
        if agg_fn is None:
            agg_fn = "mean" if ("rate" in title.lower() or "%" in title) else "count"

        pt = (
            data.groupby(["creation_day_name", "creation_hour"])
                .agg(value=(value_col, agg_fn))
                .reset_index()
        )
        pt["label"] = pt["value"].map(label_fmt)

        z = (
            pt.pivot(index="creation_day_name",
                     columns="creation_hour",
                     values="value")
            .reindex(index=WEEKDAYS, columns=hour_order)
            .fillna(0)                              # safety‑net for any gaps
        )
        text = (
            pt.pivot(index="creation_day_name",
                     columns="creation_hour",
                     values="label")
            .reindex(index=WEEKDAYS, columns=hour_order)
            .fillna("0")
        )

        fig = go.Figure(
            go.Heatmap(
                z=z.values,
                x=hour_order,
                y=WEEKDAYS,
                text=text.values,
                texttemplate="%{text}",
                textfont=dict(color="black", size=9),
                colorscale=colorscale,
                colorbar=dict(tickformat=zformat),
            )
        )
        fig.update_layout(
            title=title,
            yaxis=dict(categoryorder="array", categoryarray=WEEKDAYS),
        )
        return fig

    # ------------------------------------------------------------------
    # 4.  Build the three charts
    # ------------------------------------------------------------------

    fig_win = make_heatmap(
        df[["creation_day_name", "creation_hour", "is_won"]],
        "is_won",
        "{:.1%}".format,
        "Blues",
        "Win %",
    )

    fig_cnt = make_heatmap(
        df[["creation_day_name", "creation_hour", "TL_id"]],
        "TL_id",
        lambda x: f"{int(x)}",
        "Greens",
        "Lead Count",
        zformat=",.0f",
    )

    fig_ttf = make_heatmap(
        df[["creation_day_name", "creation_hour", "CreationToFirstAction_min"]]
        .assign(CreationToFirstAction_min=lambda d: d["CreationToFirstAction_min"].fillna(0)),
        "CreationToFirstAction_min",
        lambda x: f"{x:.0f}",          # show whole‑minute values
        "Purples",
        "Time to First Action (min)",
        zformat=",.0f",
        agg_fn="median"
    )

    fig_con = make_heatmap(
        df[["creation_day_name", "creation_hour", "is_contacted"]],
        "is_contacted",
        "{:.1%}".format,
        "OrRd",
        "Contact %",
    )

    st.plotly_chart(fig_win, use_container_width=True)
    st.plotly_chart(fig_cnt, use_container_width=True)
    st.plotly_chart(fig_ttf, use_container_width=True)
    st.plotly_chart(fig_con, use_container_width=True)




def page_calls(df):
    st.subheader("Call volume vs Wins")
    if "calls_out" not in df.columns:
        st.info("Call log not merged yet.")
        return
    agg = (
        df.groupby(df["Creation Date"].dt.date)
          .agg(calls=("calls_out","sum"), wins=("is_won","sum"))
          .reset_index()
          .rename(columns={"Creation Date":"date"})
    )
    fig = px.bar(agg, x="date", y="calls", title="Outbound Calls")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(agg, x="date", y="wins", title="Wins", markers=True)
    st.plotly_chart(fig2, use_container_width=True)

def page_quality(df):
    st.subheader("Data Quality – missing critical fields")
    crit = ["Phone", "Email", "CreationToFirstAction_min"]
    miss = {c: df[c].isna().mean() for c in crit}
    miss_df = pd.DataFrame({"Field": miss.keys(), "MissingPct": miss.values()})
    fig = px.bar(miss_df, x="Field", y="MissingPct")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    show_table(miss_df, pct_cols=["MissingPct"])

def page_geo(df):
    df["Zip5"] = df["Zip5"].astype(str).str.zfill(5)
    # ────────────────────────  STATE‑LEVEL SUMMARY  ──────────────────────────
    st.subheader("State‑level summary")

    state_tbl = (
        df.groupby("State")                                 # ← aggregate by state
          .agg(
              Lead_count   = ("TL_id",        "size"),
              win_count=("is_won", "sum"),
              contact_rate = ("is_contacted", "mean"),
              quote_rate   = ("is_quoted",    "mean"),
              close_rate   = ("is_won",       "mean")
          )
          .reset_index()
          .sort_values("win_count", ascending=False)
    )

    # show_table already formats % columns automatically
    show_table(state_tbl, pct_cols=["contact_rate",
                                    "quote_rate",
                                    "close_rate"],
               number_cols=['Lead_count','win_count'])

    st.markdown("---")   # visual break before the existing ZIP map

    # ────────────────────────  CITY‑LEVEL SUMMARY  ──────────────────────────
    st.subheader("City‑level summary")

    city_tbl = (
        df.groupby("City")                                 # ← aggregate by state
          .agg(
              Lead_count   = ("TL_id",        "size"),
              win_count=("is_won", "sum"),
              contact_rate = ("is_contacted", "mean"),
              quote_rate   = ("is_quoted",    "mean"),
              close_rate   = ("is_won",       "mean")
          )
          .reset_index()
          .sort_values("win_count", ascending=False)
    )

    # show_table already formats % columns automatically
    show_table(city_tbl, pct_cols=["contact_rate",
                                    "quote_rate",
                                    "close_rate"],
               number_cols=['Lead_count','win_count'])

    st.markdown("---")
    # ────────────────────────  SEGMENT‑LEVEL SUMMARY  ──────────────────────────
    st.subheader("Segment‑level summary")

    segment_tbl = (
        df.groupby(["order","SegmentName"])                                 # ← aggregate by segment
          .agg(
              Lead_count   = ("TL_id",        "size"),
              win_count=("is_won", "sum"),
              contact_rate = ("is_contacted", "mean"),
              quote_rate   = ("is_quoted",    "mean"),
              close_rate   = ("is_won",       "mean")
          )
          .reset_index()
          .sort_values("win_count", ascending=False)
    )

    # show_table already formats % columns automatically
    show_table(segment_tbl, pct_cols=["contact_rate",
                                    "quote_rate",
                                    "close_rate"],
               number_cols=['order','Lead_count','win_count'])

    st.markdown("---")

    st.subheader("ZIP‑level heat‑map")

    # Metric selector
    metric = st.selectbox(
        "Colour by …", ["Lead count", "Win rate ( %)"], index=0
    )

    agg = (
        df.groupby("Zip5")
          .agg(leads=("TL_id", "size"),
               win_rate=("is_won", "mean"))
          .reset_index()
    )
    if metric.startswith("Lead"):
        agg["value"] = agg["leads"]
        color_scale = "Viridis"
        fmt = ""
    else:
        agg["value"] = agg["win_rate"]
        color_scale = "Plasma"
        fmt = ".0%"


    fig = px.choropleth_map(
        agg,
        geojson="https://raw.githubusercontent.com/OpenDataDE/State-zip-code-GeoJSON/master/zip_codes_geo.json",
        locations="Zip5",
        featureidkey="properties.ZCTA5CE10",
        color="value",
        color_continuous_scale=color_scale,
        map_style="carto-positron",  # ← change here
        zoom=3,
        center={"lat": 37.1, "lon": -95.7},
        opacity=0.6,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar_tickformat=fmt,
    )
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------
# 5.  Render selected page
# ------------------------------------------------------------------
globals()[PAGES[page]](df)

