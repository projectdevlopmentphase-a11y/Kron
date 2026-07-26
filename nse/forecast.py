"""Forecast NSE equity candles with the Kronos foundation model.

Two data sources are supported:
  --csv <path>              use a local OHLCV CSV (columns: timestamps, open,
                             high, low, close, volume[, amount])
  --symbol <tradingsymbol>  pull live daily candles from Kite Connect
                             (requires KITE_API_KEY / KITE_ACCESS_TOKEN)

By default the last `--lookback` candles are used as context and Kronos
forecasts the `--pred-len` candles that come *after* them into the future
(the "--future" mode, which is the default).

Pass `--backtest` to instead hold out the most recent `--pred-len` *known*
candles and score Kronos against what actually happened: it re-forecasts
one day at a time, feeding each day's real outcome back in as context
before predicting the next day (a "re-run every morning with last night's
real close" walk-forward simulation), and reports MAE / RMSE / MAPE on
close.

Example (forecast the next 20 trading days for HDFCBANK from a local CSV):
  python -m nse.forecast --csv data/NSE_HDFCBANK_day.csv --lookback 400 \\
      --pred-len 20 --output data/NSE_HDFCBANK_forecast.csv \\
      --chart-output data/NSE_HDFCBANK_forecast.png

Example (backtest against the most recent 20 known trading days, marking
HDFC Bank's Q1 FY27 results on the chart):
  python -m nse.forecast --csv data/NSE_HDFCBANK_day.csv --lookback 400 \\
      --pred-len 20 --backtest --event "2026-07-18:Q1 FY27 results" \\
      --output data/NSE_HDFCBANK_backtest.csv \\
      --chart-output data/NSE_HDFCBANK_backtest.png
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kronos.model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402

TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_NAME = "NeoQuasar/Kronos-small"
PRICE_COLS = ["open", "high", "low", "close", "volume", "amount"]


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


def future_business_days(last_timestamp: pd.Timestamp, pred_len: int) -> pd.Series:
    dates = pd.bdate_range(start=last_timestamp, periods=pred_len + 1, freq="B")[1:]
    return pd.Series(dates)


def parse_events(raw_events):
    events = []
    for raw in raw_events or []:
        date_str, _, label = raw.partition(":")
        events.append((pd.Timestamp(date_str), label or date_str))
    return events


def annotate_events(ax, events):
    for i, (event_date, label) in enumerate(events):
        ax.axvline(event_date, color="#9467bd", linestyle="-.", linewidth=1.5, zorder=0)
        ax.annotate(
            label,
            xy=(event_date, 1),
            xycoords=("data", "axes fraction"),
            xytext=(4, -10 - 12 * (i % 3)),
            textcoords="offset points",
            fontsize=8,
            color="#9467bd",
            rotation=90,
            va="top",
        )


def plot_forecast(history_df, pred_df, symbol, chart_output, events=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(history_df["timestamps"], history_df["close"], label="Historical close", color="#1f77b4")

    connector_x = [history_df["timestamps"].iloc[-1]] + list(pred_df.index)
    connector_y = [history_df["close"].iloc[-1]] + list(pred_df["close"])
    ax.plot(connector_x, connector_y, label="Kronos forecast", color="#d62728", linestyle="--", marker="o", markersize=3)

    ax.axvline(history_df["timestamps"].iloc[-1], color="gray", linestyle=":", linewidth=1)
    if events:
        annotate_events(ax, events)
    ax.set_title(f"{symbol} — close price, historical vs Kronos forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(chart_output, dpi=150)
    plt.close(fig)


def walk_forward_backtest(df, lookback, pred_len, predictor, T, top_p, sample_count, intraday_df=None, intraday_tail=0):
    """Re-forecast one day at a time, feeding each day's *actual* outcome back in
    as context before predicting the next day (a realistic "re-run every morning
    with last night's real close" simulation). If intraday_df is given, the last
    `intraday_tail` intraday candles of the prior trading day replace that day's
    single daily bar as the most recent context (same idea as --append-csv, but
    repeated at every backtest step).
    """
    n = len(df)
    start = n - pred_len
    rows = []
    for i in range(pred_len):
        idx = start + i
        context = df.iloc[idx - lookback : idx]
        day_intraday = pd.DataFrame()
        if intraday_df is not None and intraday_tail > 0:
            prev_day = df.iloc[idx - 1]["timestamps"].normalize()
            day_intraday = intraday_df[intraday_df["timestamps"].dt.normalize() == prev_day].tail(intraday_tail)
            if len(day_intraday):
                context = pd.concat([context.iloc[len(day_intraday):], day_intraday], ignore_index=True)
        target_timestamp = df.iloc[idx]["timestamps"]
        y_timestamp = pd.Series([target_timestamp])
        pred = predictor.predict(
            df=context[PRICE_COLS],
            x_timestamp=context["timestamps"],
            y_timestamp=y_timestamp,
            pred_len=1,
            T=T,
            top_p=top_p,
            sample_count=sample_count,
            verbose=False,
        )
        pred.index = y_timestamp.values
        rows.append(pred)
        tag = " (+intraday)" if len(day_intraday) else ""
        print(f"  {target_timestamp.date()}: predicted close={pred['close'].iloc[0]:.2f}{tag}")
    return pd.concat(rows)


def compute_metrics(actual_df, pred_df, column="close"):
    actual = actual_df[column].to_numpy()
    predicted = pred_df[column].to_numpy()
    errors = predicted - actual
    mae = abs(errors).mean()
    rmse = (errors**2).mean() ** 0.5
    mape = (abs(errors) / actual).mean() * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE_pct": mape}


def plot_backtest(context_df, actual_df, pred_df, symbol, chart_output, events=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    context_tail = context_df.tail(60)
    ax.plot(context_tail["timestamps"], context_tail["close"], label="Context (input)", color="#1f77b4")

    actual_connector_x = [context_df["timestamps"].iloc[-1]] + list(actual_df["timestamps"])
    actual_connector_y = [context_df["close"].iloc[-1]] + list(actual_df["close"])
    ax.plot(actual_connector_x, actual_connector_y, label="Actual close", color="#2ca02c", marker="o", markersize=4)

    pred_connector_x = [context_df["timestamps"].iloc[-1]] + list(pred_df.index)
    pred_connector_y = [context_df["close"].iloc[-1]] + list(pred_df["close"])
    ax.plot(pred_connector_x, pred_connector_y, label="Kronos forecast (walk-forward)", color="#d62728", linestyle="--", marker="o", markersize=4)

    ax.axvline(context_df["timestamps"].iloc[-1], color="gray", linestyle=":", linewidth=1)
    if events:
        annotate_events(ax, events)
    ax.set_title(f"{symbol} — backtest: Kronos forecast vs actual close")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(chart_output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    parser.add_argument("--chart-output", help="path to write a historical-vs-forecast PNG chart to")
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="hold out the most recent pred-len known candles and walk-forward score "
        "Kronos against what actually happened, instead of forecasting real future "
        "dates (the default)",
    )
    parser.add_argument(
        "--event",
        action="append",
        help="mark a real-world event (e.g. an earnings date) on the chart as "
        "YYYY-MM-DD:label; repeatable",
    )
    parser.add_argument(
        "--append-csv",
        help="path to a finer-grained OHLCV CSV (e.g. intraday 60minute candles) "
        "whose most recent --append-tail rows are appended as the most recent "
        "context. In --future mode this is done once, after the daily lookback "
        "tail; in --backtest mode it's done at every walk-forward step, replacing "
        "each held-out day's single daily bar with that prior day's own intraday "
        "candles (matched by date)",
    )
    parser.add_argument(
        "--append-tail",
        type=int,
        default=2,
        help="how many most-recent rows of --append-csv to append (default 2, e.g. "
        "the last 2 hours of 60minute candles)",
    )
    args = parser.parse_args()
    events = parse_events(args.event)

    if not args.csv and not args.symbol:
        parser.error("either --csv or --symbol is required")

    df = load_dataframe(args)
    symbol = args.symbol or Path(args.csv).stem
    lookback, pred_len = args.lookback, args.pred_len

    if len(df) < lookback + (pred_len if args.backtest else 0):
        raise SystemExit(
            f"Need at least {lookback + (pred_len if args.backtest else 0)} candles, got {len(df)}. "
            "Widen --from-date/--to-date or lower --lookback/--pred-len."
        )

    print(f"Loading {MODEL_NAME} / {TOKENIZER_NAME} ...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
    model = Kronos.from_pretrained(MODEL_NAME)
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    if args.backtest:
        window = df.tail(lookback + pred_len).reset_index(drop=True)
        context_df = window.iloc[:lookback].reset_index(drop=True)
        actual_df = window.iloc[lookback:].reset_index(drop=True)
        y_timestamp = actual_df["timestamps"]

        intraday_df = None
        if args.append_csv:
            intraday_df = pd.read_csv(args.append_csv)
            intraday_df["timestamps"] = pd.to_datetime(intraday_df["timestamps"])
            print(f"Using intraday candles from {args.append_csv} at each walk-forward step (last {args.append_tail} per prior day)")

        print(f"Backtesting {pred_len} sessions, walk-forward from {lookback} candles of history ...")
        pred_df = walk_forward_backtest(
            df, lookback, pred_len, predictor, args.temperature, args.top_p, args.sample_count,
            intraday_df=intraday_df, intraday_tail=args.append_tail,
        )

        print("\nForecast:")
        print(pred_df)

        metrics = compute_metrics(actual_df, pred_df, column="close")
        comparison = pd.DataFrame(
            {
                "actual_close": actual_df["close"].values,
                "predicted_close": pred_df["close"].values,
                "error": pred_df["close"].values - actual_df["close"].values,
            },
            index=y_timestamp.values,
        )
        print("\nActual vs predicted (close):")
        print(comparison)
        print(
            f"\nClose-price accuracy over {pred_len} sessions: "
            f"MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}  MAPE={metrics['MAPE_pct']:.2f}%"
        )

        for event_date, label in events:
            before = comparison[comparison.index < event_date]
            after = comparison[comparison.index >= event_date]
            if len(before) and len(after):
                print(
                    f"\nAround event '{label}' ({event_date.date()}): "
                    f"MAE before={before['error'].abs().mean():.2f} ({len(before)} sessions), "
                    f"MAE after={after['error'].abs().mean():.2f} ({len(after)} sessions)"
                )

        if args.output:
            comparison.to_csv(args.output)
            print(f"\nWrote backtest comparison to {args.output}")

        if args.chart_output:
            plot_backtest(context_df, actual_df, pred_df, symbol, args.chart_output, events=events)
            print(f"Wrote chart to {args.chart_output}")
    else:
        history_for_plot = df
        if args.append_csv:
            extra = pd.read_csv(args.append_csv)
            extra["timestamps"] = pd.to_datetime(extra["timestamps"])
            extra = extra.tail(args.append_tail).reset_index(drop=True)
            tail = df.tail(lookback - len(extra)).reset_index(drop=True)
            tail = pd.concat([tail, extra], ignore_index=True)
            history_for_plot = pd.concat([df, extra], ignore_index=True)
            print(
                f"Appended {len(extra)} rows from {args.append_csv} "
                f"({extra['timestamps'].iloc[0]} .. {extra['timestamps'].iloc[-1]}) "
                "as the most recent context"
            )
        else:
            tail = df.tail(lookback).reset_index(drop=True)

        x_df = tail[PRICE_COLS]
        x_timestamp = tail["timestamps"]
        y_timestamp = future_business_days(tail["timestamps"].iloc[-1], pred_len)

        print(f"Forecasting {pred_len} future candles from {len(tail)} candles of history ...")
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

        if args.chart_output:
            plot_forecast(history_for_plot, pred_df, symbol, args.chart_output, events=events)
            print(f"Wrote chart to {args.chart_output}")


if __name__ == "__main__":
    main()
