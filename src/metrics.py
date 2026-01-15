import numpy as np
import pandas as pd


def infer_step_order(process_steps: pd.Series) -> dict:
    """
    Infer step order mapping from known canonical steps.
    If your dataset uses different labels, edit the 'canonical' list.
    """
    canonical = ["start", "step_1", "step_2", "step_3", "confirm"]
    existing = [s for s in canonical if s in set(process_steps)]
    return {s: i for i, s in enumerate(existing)}


def _has_backtrack(step_orders: pd.Series) -> int:
    arr = step_orders.dropna().to_numpy()
    if len(arr) < 2:
        return 0
    return int(np.any(np.diff(arr) < 0))


def sessionize(events: pd.DataFrame, step_order: dict) -> pd.DataFrame:
    """
    Convert event-level web data into session-level table (1 row per visit_id).

    Outputs:
      - completed (has confirm)
      - total_time_sec (start -> confirm if completed else start -> last event)
      - n_events, n_unique_steps
      - loop_like, backtrack
    """
    df = events.copy()
    df["step_order"] = df["process_step"].map(step_order)
    df = df.sort_values(["visit_id", "date_time"])

    # completed if confirm exists
    has_confirm = df.groupby("visit_id")["process_step"].apply(lambda s: (s == "confirm").any())
    confirm_time = df[df["process_step"] == "confirm"].groupby("visit_id")["date_time"].min()

    sessions = df.groupby(["visit_id", "client_id", "Variation"], as_index=False).agg(
        start_time=("date_time", "min"),
        end_time=("date_time", "max"),
        n_events=("process_step", "size"),
        n_unique_steps=("process_step", "nunique"),
    )

    sessions["completed"] = sessions["visit_id"].map(has_confirm).fillna(False).astype(int)
    sessions["confirm_time"] = sessions["visit_id"].map(confirm_time)

    sessions["finish_time"] = sessions["confirm_time"].fillna(sessions["end_time"])
    sessions["total_time_sec"] = (sessions["finish_time"] - sessions["start_time"]).dt.total_seconds()

    # proxy "error/friction": lots of events but few unique steps
    sessions["loop_like"] = ((sessions["n_events"] >= 8) & (sessions["n_unique_steps"] <= 3)).astype(int)

    # backtracking
    backtrack = df.groupby("visit_id")["step_order"].apply(_has_backtrack)
    sessions["backtrack"] = sessions["visit_id"].map(backtrack).fillna(0).astype(int)

    return sessions


def kpis_by_group(sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Compute core KPIs per variation:
      - completion_rate
      - median + p90 time
      - loop_rate, backtrack_rate
    """
    out = sessions.groupby("Variation").agg(
        sessions=("visit_id", "nunique"),
        completion_rate=("completed", "mean"),
        median_time_sec=("total_time_sec", "median"),
        p90_time_sec=("total_time_sec", lambda x: np.nanpercentile(x, 90)),
        loop_rate=("loop_like", "mean"),
        backtrack_rate=("backtrack", "mean"),
    ).reset_index()

    return out
