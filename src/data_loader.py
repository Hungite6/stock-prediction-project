"""
data_loader.py
Load stock data from Yahoo Finance and save to data/raw/
"""

import yfinance as yf
import pandas as pd
import os


def load_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download stock data from Yahoo Finance.

    Args:
        ticker: e.g. "FPT.VN", "VNM.VN"
        start:  "YYYY-MM-DD"
        end:    "YYYY-MM-DD"

    Returns:
        DataFrame with OHLCV columns
    """
    print(f"[DataLoader] Downloading {ticker} from {start} to {end} ...")
    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    # Flatten MultiIndex columns if present (yfinance sometimes returns them)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print(f"[DataLoader] Downloaded {len(df)} rows. Columns: {list(df.columns)}")
    return df


def save_raw_data(df: pd.DataFrame, ticker: str, save_dir: str = "data/raw") -> str:
    """
    Save raw DataFrame to CSV.

    Returns:
        Path to saved file
    """
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{ticker.replace('.', '_')}_raw.csv"
    path = os.path.join(save_dir, filename)
    df.to_csv(path)
    print(f"[DataLoader] Saved raw data → {path}")
    return path


def load_raw_data(ticker: str, save_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load previously saved raw CSV.
    """
    filename = f"{ticker.replace('.', '_')}_raw.csv"
    path = os.path.join(save_dir, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data not found: {path}. Run load_stock_data() first.")

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"[DataLoader] Loaded {len(df)} rows from {path}")
    return df