import pandas as pd
import numpy as np
from typing import Literal, Optional
import pytz



class CallDataAnalyzer:
    """
    Analyze 8x8 Work CDR data.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw rows from fetch_cdrs().  Expected columns are listed in the docstring
        of summarize().
    tz : str, default "US/Eastern"
        IANA time‑zone name to convert all timestamps to.
    """

    _REQUIRED_COLUMNS = {
        "direction", "answered", "talkTimeMS",
        "ringDuration", "abandonedTime",
        "branches", "caller", "callerName",
        "callee", "calleeName",
        "startTimeUTC", "startTime"
    }

    def __init__(self, df: pd.DataFrame, tz: str = "US/Eastern") -> None:
        # ---------- unchanged pre‑flight checks ----------
        missing = self._REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        self.tz = pytz.timezone(tz)
        self.df = df.copy()

        # ---------- NEW: identify 8x8 employee ----------
        # Normalize direction to lower case once
        self.df["direction_lc"] = self.df["direction"].str.lower()

        # Employee extension / name depending on direction
        self.df["emp_number"] = np.where(
            self.df["direction_lc"] == "incoming",
            self.df["callee"],
            self.df["caller"]
        )
        self.df["emp_name"] = np.where(
            self.df["direction_lc"] == "incoming",
            self.df["calleeName"],
            self.df["callerName"]
        )
        self.df["counterparty_number"] = np.where(
            self.df["direction_lc"] == "incoming",
            self.df["caller"],
            self.df["callee"]
        )

        self.df["counterparty_name"] = np.where(
            self.df["direction_lc"] == "incoming",
            self.df["callerName"],
            self.df["callee"]
        )

        # --- normalize times -------------------------------------------------
        # Prefer the explicit *UTC* column if present; fall back to local.
        for col in ("startTimeUTC", "connectTimeUTC", "disconnectedTimeUTC"):
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], unit="ms", utc=True)
                # Create a local‑time sibling with _ET suffix
                self.df[col.replace("UTC", "ET")] = (
                    self.df[col].dt.tz_convert(self.tz)
                )

        # --- derive helper columns ------------------------------------------
        self.df["minutes"] = self.df["talkTimeMS"].fillna(0) / 1000 / 60
        self.df["callTime_min"] = df["callTime"] / 60000  # 1000 ms/sec * 60 sec/min
        self.df["ringDuration_min"] = df["ringDuration"] / 60000
        self.df["abandonedTime_min"] = df["abandonedTime"] / 60000
        self.df["answered_flag"] = self.df["answered"].str.lower().eq("answered")
        self.df["is_outbound"] = self.df["direction_lc"] == "outgoing"
        self.df["is_inbound"] = self.df["direction_lc"] == "incoming"

        self.df["branch_key"] = self.df["branches"].apply(
            lambda x: ";".join(sorted(set(x))) if isinstance(x, list) else str(x)
        )

    # ------------------------------------------------------------------ #
    #  summarize() – now by branch_key + employee                         #
    # ------------------------------------------------------------------ #
    def summarize(self) -> pd.DataFrame:
        """
        One row per branch_key + employee number + employee name.

        Columns returned (same metrics as before):
          branch_key, emp_number, emp_name,
          total_calls, total_outbound_calls, total_inbound_calls,
          outbound_answer_rate_pct, inbound_answer_rate_pct,
          total_minutes, total_outbound_minutes, total_inbound_minutes,
          avg_minutes, avg_minutes_inbound, avg_minutes_outbound,
          total_ring_duration, avg_ring_duration,
          total_abandoned_time, avg_abandoned_time
        """
        gcols = ["branch_key", "emp_number", "emp_name"]
        g = self.df.groupby(gcols, dropna=False)

        def _pct(n, d):
            return np.where(d == 0, 0.0, n / d * 100.0)

        summary = g.agg(
            total_calls=("callId", "count"),
            total_outbound_calls=("is_outbound", "sum"),
            total_inbound_calls=("is_inbound", "sum"),
            answered_outbound=("answered_flag",
                               lambda s: ((s & self.df.loc[s.index, "is_outbound"]).sum())),
            answered_inbound=("answered_flag",
                              lambda s: ((s & self.df.loc[s.index, "is_inbound"]).sum())),
            total_minutes=("minutes", "sum"),
            total_outbound_minutes=("minutes",
                                    lambda m: m[self.df.loc[m.index, "is_outbound"]].sum()),
            total_inbound_minutes=("minutes",
                                   lambda m: m[self.df.loc[m.index, "is_inbound"]].sum()),
            avg_minutes=("minutes", "mean"),
            avg_minutes_outbound=("minutes",
                                  lambda m: m[self.df.loc[m.index, "is_outbound"]].mean()),
            avg_minutes_inbound=("minutes",
                                 lambda m: m[self.df.loc[m.index, "is_inbound"]].mean()),
            total_ring_duration=("ringDuration", "sum"),
            avg_ring_duration=("ringDuration", "mean"),
            total_abandoned_time=("abandonedTime", "sum"),
            avg_abandoned_time=("abandonedTime", "mean"),
        ).reset_index()

        summary["outbound_answer_rate_pct"] = _pct(
            summary["answered_outbound"], summary["total_outbound_calls"]
        )
        summary["inbound_answer_rate_pct"] = _pct(
            summary["answered_inbound"], summary["total_inbound_calls"]
        )

        return (summary
                .drop(columns=["answered_outbound", "answered_inbound"])
                .sort_values(gcols))

    def summarize_branch(self) -> pd.DataFrame:
        """
        Return one row per **branch_key** with the following columns:

        ------------------------------------------------------------------
        branch_key, total_calls,
        total_outbound_calls, total_inbound_calls,
        outbound_answer_rate_pct, inbound_answer_rate_pct,
        total_minutes, total_outbound_minutes, total_inbound_minutes,
        avg_minutes, avg_minutes_inbound, avg_minutes_outbound,
        total_ring_duration, avg_ring_duration,
        total_abandoned_time, avg_abandoned_time
        ------------------------------------------------------------------
        """
        g = self.df.groupby(["branch_key"], dropna=False)

        def _pct(numer, denom):
            return np.where(denom == 0, 0.0, numer / denom * 100.0)

        summary = g.agg(
            total_employees=("emp_number", "nunique"),
            total_counterparties=("counterparty_number", "nunique"),
            total_calls=("callId", "count"),
            total_outbound_calls=("is_outbound", "sum"),
            total_inbound_calls=("is_inbound", "sum"),
            answered_outbound=("answered_flag", lambda s: ((s & self.df.loc[s.index, "is_outbound"]).sum())),
            answered_inbound=("answered_flag", lambda s: ((s & self.df.loc[s.index, "is_inbound"]).sum())),
            total_minutes=("minutes", "sum"),
            total_outbound_minutes=("minutes", lambda m: m[self.df.loc[m.index, "is_outbound"]].sum()),
            total_inbound_minutes=("minutes", lambda m: m[self.df.loc[m.index, "is_inbound"]].sum()),
            avg_minutes=("minutes", "mean"),
            avg_minutes_outbound=("minutes", lambda m: m[self.df.loc[m.index, "is_outbound"]].mean()),
            avg_minutes_inbound=("minutes", lambda m: m[self.df.loc[m.index, "is_inbound"]].mean()),
            total_ring_duration=("ringDuration", "sum"),
            avg_ring_duration=("ringDuration", "mean"),
            total_abandoned_time=("abandonedTime", "sum"),
            avg_abandoned_time=("abandonedTime", "mean"),
        ).reset_index()

        # Answer‑rate percentages
        summary["outbound_answer_rate_pct"] = _pct(
            summary["answered_outbound"], summary["total_outbound_calls"]
        )
        summary["inbound_answer_rate_pct"] = _pct(
            summary["answered_inbound"], summary["total_inbound_calls"]
        )

        # drop helper columns we no longer need
        summary = summary.drop(columns=["answered_outbound", "answered_inbound"])
        return summary.sort_values(["total_minutes"], ascending=[False])

    # ------------------------------------------------------------------ #
    #  summarize() – now by branch_key + employee                         #
    # ------------------------------------------------------------------ #
    def summarize_counterparty(self) -> pd.DataFrame:
        """
        One row per counterparty number + counterparty name.

        Columns returned (same metrics as before):
          counterparty_number,counterparty_name,
          total_calls, total_outbound_calls, total_inbound_calls,
          outbound_answer_rate_pct, inbound_answer_rate_pct,
          total_minutes, total_outbound_minutes, total_inbound_minutes,
          avg_minutes, avg_minutes_inbound, avg_minutes_outbound,
          total_ring_duration, avg_ring_duration,
          total_abandoned_time, avg_abandoned_time
        """
        gcols = ["counterparty_number", "counterparty_name"]
        g = self.df.groupby(gcols, dropna=False)

        def _pct(n, d):
            return np.where(d == 0, 0.0, n / d * 100.0)

        summary = g.agg(
            total_calls=("callId", "count"),
            total_outbound_calls=("is_outbound", "sum"),
            total_inbound_calls=("is_inbound", "sum"),
            answered_outbound=("answered_flag",
                               lambda s: ((s & self.df.loc[s.index, "is_outbound"]).sum())),
            answered_inbound=("answered_flag",
                              lambda s: ((s & self.df.loc[s.index, "is_inbound"]).sum())),
            total_minutes=("minutes", "sum"),
            total_outbound_minutes=("minutes",
                                    lambda m: m[self.df.loc[m.index, "is_outbound"]].sum()),
            total_inbound_minutes=("minutes",
                                   lambda m: m[self.df.loc[m.index, "is_inbound"]].sum()),
            avg_minutes=("minutes", "mean"),
            avg_minutes_outbound=("minutes",
                                  lambda m: m[self.df.loc[m.index, "is_outbound"]].mean()),
            avg_minutes_inbound=("minutes",
                                 lambda m: m[self.df.loc[m.index, "is_inbound"]].mean()),
            total_ring_duration=("ringDuration", "sum"),
            avg_ring_duration=("ringDuration", "mean"),
            total_abandoned_time=("abandonedTime", "sum"),
            avg_abandoned_time=("abandonedTime", "mean"),
        ).reset_index()

        summary["outbound_answer_rate_pct"] = _pct(
            summary["answered_outbound"], summary["total_outbound_calls"]
        )
        summary["inbound_answer_rate_pct"] = _pct(
            summary["answered_inbound"], summary["total_inbound_calls"]
        )

        return (summary
                .drop(columns=["answered_outbound", "answered_inbound"])
                .sort_values(gcols))

        # --------------------------------------------------------------------- #

    #  TEMP‑OR‑AL  (hour‑of‑day × day‑of‑week)                              #
    # --------------------------------------------------------------------- #
    def temporal_grid(
            self,
            metric: Literal[
                "call_count",
                "answer_rate_out",
                "answer_rate_in",
                "avg_minutes",
            ] = "call_count",
            day_start_hour: int = 0,
            day_end_hour: int = 23,
            inbound: Optional[bool] = None,
            outbound: Optional[bool] = None,
    ) -> pd.DataFrame:
        """
        Hour‑of‑day × day‑of‑week matrix.

        Parameters
        ----------
        metric
            "call_count" | "answer_rate_out" | "answer_rate_in" | "avg_minutes"
        day_start_hour, day_end_hour
            Inclusive hour range (0‑23) to keep.
        inbound, outbound
            Pass True/False to force one direction; leave both None for all calls.

        Returns
        -------
        pandas.DataFrame  (rows = 0‑23, cols = Mon…Sun, values = metric chosen)
        """

        if inbound and outbound:
            raise ValueError("Choose inbound OR outbound, not both True.")

        df = self.df.copy()

        # direction filter ----------------------------------------------------
        if inbound is True:
            df = df[df["is_inbound"]]
        elif outbound is True:
            df = df[df["is_outbound"]]

        # local‑time timestamp -----------------------------------------------
        if "startTimeET" in df.columns:
            ts = df["startTimeET"]
        else:
            ts = pd.to_datetime(df["startTimeUTC"], utc=True).dt.tz_convert(self.tz)

        df["hour"] = ts.dt.hour
        df["dow"] = ts.dt.dayofweek  # Monday = 0
        dow_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        df["dow_name"] = df["dow"].map(dow_map)

        # helper masks --------------------------------------------------------
        out_mask = df["is_outbound"]
        in_mask = df["is_inbound"]
        ans_mask = df["answered_flag"]

        # metric‑specific pivot ----------------------------------------------
        if metric == "call_count":
            pivot = pd.pivot_table(
                df, index="hour", columns="dow_name", values="callId",
                aggfunc="count", fill_value=0
            )

        elif metric == "avg_minutes":
            pivot = pd.pivot_table(
                df, index="hour", columns="dow_name", values="minutes",
                aggfunc="mean"
            )

        elif metric in ("answer_rate_out", "answer_rate_in"):
            use_mask = out_mask if metric.endswith("_out") else in_mask
            # denominator: total directional calls
            denom = pd.pivot_table(
                df[use_mask], index="hour", columns="dow_name", values="callId",
                aggfunc="count", fill_value=0
            )
            # numerator: answered *and* directional
            numer = pd.pivot_table(
                df[use_mask & ans_mask], index="hour", columns="dow_name",
                values="callId", aggfunc="count", fill_value=0
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                pivot = numer.divide(denom).multiply(100.0)
            pivot.replace([np.inf, -np.inf], np.nan, inplace=True)

        else:
            raise ValueError(f"Unknown metric: {metric}")

        # tidy index/column order --------------------------------------------
        hour_range = range(day_start_hour, day_end_hour + 1)
        day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        pivot = (pivot
                 .reindex(index=hour_range, fill_value=0 if metric == "call_count" else np.nan)
                 .reindex(columns=day_order, fill_value=0 if metric == "call_count" else np.nan)
                 )

        return pivot

# ---------------------  Example usage  ---------------------
# df = pd.DataFrame(rows)  # rows from fetch_cdrs()
# analyzer = CallDataAnalyzer(df)
# branch_caller_summary = analyzer.summarize()
# grid = analyzer.temporal_grid(metric="answer_rate_out", outbound=True)
