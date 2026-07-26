"""Thin wrapper around kiteconnect for pulling NSE equity OHLCV candles.

Credentials are read from the environment:
  KITE_API_KEY      - Kite Connect app API key
  KITE_ACCESS_TOKEN - Access token from a completed Kite Connect login

Both must be set before using KiteHistoricalDataClient.
"""
import os

import pandas as pd
from kiteconnect import KiteConnect


class KiteHistoricalDataClient:
    def __init__(self, api_key: str = None, access_token: str = None):
        api_key = api_key or os.environ.get("KITE_API_KEY")
        access_token = access_token or os.environ.get("KITE_ACCESS_TOKEN")
        if not api_key or not access_token:
            raise RuntimeError(
                "KITE_API_KEY and KITE_ACCESS_TOKEN must be set to fetch live data"
            )
        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)
        self._instrument_cache = None

    def _instruments(self, exchange: str = "NSE"):
        if self._instrument_cache is None:
            self._instrument_cache = self.kite.instruments(exchange)
        return self._instrument_cache

    def get_instrument_token(self, tradingsymbol: str, exchange: str = "NSE") -> int:
        for row in self._instruments(exchange):
            if row["tradingsymbol"] == tradingsymbol and row["segment"] == exchange:
                return row["instrument_token"]
        raise ValueError(f"{exchange}:{tradingsymbol} not found in instrument dump")

    def get_historical_ohlcv(
        self,
        tradingsymbol: str,
        from_date: str,
        to_date: str,
        interval: str = "day",
        exchange: str = "NSE",
    ) -> pd.DataFrame:
        """Returns a DataFrame with columns:
        timestamps, open, high, low, close, volume, amount
        (amount is derived as close * volume; Kite does not provide it directly).
        """
        token = self.get_instrument_token(tradingsymbol, exchange)
        candles = self.kite.historical_data(token, from_date, to_date, interval)
        df = pd.DataFrame(candles)
        df = df.rename(columns={"date": "timestamps"})
        df["timestamps"] = pd.to_datetime(df["timestamps"]).dt.tz_localize(None)
        df["amount"] = df["close"] * df["volume"]
        return df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]
