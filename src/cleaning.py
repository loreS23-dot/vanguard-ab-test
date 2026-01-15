import pandas as pd


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def parse_datetime(df: pd.DataFrame, col: str = "date_time") -> pd.DataFrame:
    """Parse a datetime column to pandas datetime."""
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def clean_web(web: pd.DataFrame) -> pd.DataFrame:
    """
    Clean web event data:
      - standardize column names
      - parse datetime
      - normalize process_step strings
      - drop rows missing essential keys
      - drop exact duplicates
      - sort by visit_id + time
    """
    web = standardize_columns(web)
    web = parse_datetime(web, "date_time")

    # normalize step labels
    web["process_step"] = web["process_step"].astype(str).str.strip().str.lower()

    # drop rows that can't be used for session metrics
    web = web.dropna(subset=["visit_id", "client_id", "process_step", "date_time"])

    # robust dedupe
    dedupe_cols = ["visit_id", "client_id", "visitor_id", "process_step", "date_time"]
    dedupe_cols = [c for c in dedupe_cols if c in web.columns]
    web = web.drop_duplicates(subset=dedupe_cols)

    web = web.sort_values(["visit_id", "date_time"])
    return web


def merge_all(demo: pd.DataFrame, exp: pd.DataFrame, web: pd.DataFrame) -> pd.DataFrame:
    """
    Merge web + experiment roster + demographics into one event-level dataframe.
    Keeps only clients present in the experiment roster (inner join).
    """
    demo = standardize_columns(demo)
    exp = standardize_columns(exp)

    # Handle possible column naming
    if "variation" in exp.columns and "Variation" not in exp.columns:
        exp = exp.rename(columns={"variation": "Variation"})

    # Minimum columns we expect
    cols_needed = ["client_id", "Variation"]
    missing = [c for c in cols_needed if c not in exp.columns]
    if missing:
        raise ValueError(f"Experiment roster missing columns: {missing}. Columns found: {list(exp.columns)}")

    df = web.merge(exp[cols_needed], on="client_id", how="inner")
    df = df.merge(demo, on="client_id", how="left")
    return df
