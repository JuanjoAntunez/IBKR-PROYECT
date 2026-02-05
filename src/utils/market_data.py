from __future__ import annotations

from typing import Optional
import pandas as pd


_ALIASES = {
    "date": ["date", "Date", "datetime", "Datetime", "timestamp", "Timestamp"],
    "open": ["open", "Open"],
    "high": ["high", "High"],
    "low": ["low", "Low"],
    "close": ["close", "Close"],
    "volume": ["volume", "Volume"],
}


def normalize_ohlcv(
    df: Optional[pd.DataFrame],
    schema: str = "lower",
    set_index: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Normalize OHLCV column names and date handling.

    Args:
        df: DataFrame with OHLCV columns in any common casing.
        schema: "lower" -> open/high/low/close/volume/date,
                "upper" -> Open/High/Low/Close/Volume/Date.
        set_index: If True, set date column as DatetimeIndex when available.
    """
    if df is None or df.empty:
        return df

    if schema not in {"lower", "upper"}:
        raise ValueError("schema must be 'lower' or 'upper'")

    df = df.copy()

    rename_map = {}
    for canonical, aliases in _ALIASES.items():
        target = canonical if schema == "lower" else canonical.capitalize()
        if target in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = target
                break

    if rename_map:
        df = df.rename(columns=rename_map)

    date_col = "date" if schema == "lower" else "Date"
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        if set_index and not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index(date_col)

    return df
