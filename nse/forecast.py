"""Forecast NSE equity candles with the Kronos foundation model.

Two data sources are supported:
  --csv <path>              use a local OHLCV CSV (columns: timestamps, open,
                             high, low, close, volume[, amount])
  --symbol <tradingsymbol>  pull live daily candles from Kite Connect
                             (requires KITE_API_KEY / KITE_ACCESS_TOKEN)

Example:
  python -m nse.forecast --symbol HDFCBANK --from 2023-01-01 --to 2026-07-24 \
      --lookback 400 --pred-len 20
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kronos.model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402

TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_NAME = "NeoQuasar/Kronos-small"


def load_dataframe(args) -> pd.DataFrame:
    if args.csv:
        df = pd.read_csv(args.csv)
        df["timestamps"] = pd.to_datetime(df["timestamps"])
        return df

    from nse.kite_client import KiteHistoricalDataClient

    client = KiteHistoricalDataClient()
    return client.get_historical_ohlcv(
        tradingsymbol=args.symbol,
        from_date=args.from_date,
        to_date=args.to_date,
        interval=args.interval,
        exchange=args.exchange,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", help="NSE tradingsymbol, e.g. HDFCBANK")
    parser.add_argument("--exchange", default="NSE")
    parser.add_argument("--interval", default="day")
    parser.add_argument("--from-date", dest="from_date")
    parser.add_argument("--to-date", dest="to_date")
    parser.add_argument("--csv", help="path to a local OHLCV CSV instead of live data")
    parser.add_argument("--lookback", type=int, default=400)
    parser.add_argument("--pred-len", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--output", help="path to write the forecast CSV to")
    args = parser.parse_args()

    if not args.csv and not args.symbol:
        parser.error("either --csv or --symbol is required")

    df = load_dataframe(args)
    if len(df) < args.lookback + args.pred_len:
        raise SystemExit(
            f"Need at least {args.lookback + args.pred_len} candles, got {len(df)}. "
            "Widen --from-date/--to-date or lower --lookback/--pred-len."
        )

    lookback = args.lookback
    pred_len = args.pred_len

    x_df = df.loc[: lookback - 1, ["open", "high", "low", "close", "volume", "amount"]]
    x_timestamp = df.loc[: lookback - 1, "timestamps"]
    y_timestamp = df.loc[lookback : lookback + pred_len - 1, "timestamps"]

    print(f"Loading {MODEL_NAME} / {TOKENIZER_NAME} ...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
    model = Kronos.from_pretrained(MODEL_NAME)
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    print(f"Forecasting {pred_len} candles from {lookback} candles of history ...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=args.temperature,
        top_p=args.top_p,
        sample_count=args.sample_count,
        verbose=True,
    )
    pred_df.index = y_timestamp.values

    print("\nForecast:")
    print(pred_df)

    if args.output:
        pred_df.to_csv(args.output)
        print(f"\nWrote forecast to {args.output}")


if __name__ == "__main__":
    main()
