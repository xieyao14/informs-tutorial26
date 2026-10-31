"""Download adjusted daily closing prices from Yahoo Finance to data/yfinance/."""

import argparse
from pathlib import Path

import yfinance as yf
from src.data import DEFAULT_TICKERS


def main() -> None:
    parser = argparse.ArgumentParser(description="Download adjusted daily prices.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--start", default="2022-12-01", help="First date, inclusive")
    parser.add_argument("--end", default="2026-01-01", help="End date, exclusive")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent / "data" / "yfinance"
    data_dir.mkdir(parents=True, exist_ok=True)
    for ticker in args.tickers:
        ticker = ticker.upper()
        path = data_dir / f"{ticker}_daily_adjusted.csv"
        if path.exists() and not args.overwrite:
            print(f"{ticker}: keeping {path}")
            continue
        prices = yf.Ticker(ticker).history(
            start=args.start, end=args.end, interval="1d", auto_adjust=False
        )
        if prices.empty or "Adj Close" not in prices:
            raise RuntimeError(f"No adjusted prices returned for {ticker}.")
        adjusted = prices["Adj Close"].dropna()
        if adjusted.empty:
            raise RuntimeError(f"No adjusted prices returned for {ticker}.")
        adjusted.index = adjusted.index.strftime("%Y-%m-%d")
        adjusted.index.name = "timestamp"
        adjusted.rename("adjusted_close").to_csv(path)
        print(f"{ticker}: saved {len(adjusted)} prices to {path}")


if __name__ == "__main__":
    main()
