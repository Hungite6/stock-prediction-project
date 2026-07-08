"""
Shared debug switches for the stock forecasting pipeline.

Set these variables directly, or override them without editing code:
    $env:STOCK_DEBUG="1"
    $env:STOCK_OVERFIT_DEBUG="1"
"""

import os


DEBUG = os.getenv("STOCK_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
OVERFIT_DEBUG = os.getenv("STOCK_OVERFIT_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
DEBUG_DIR = os.path.join("reports", "debug")

