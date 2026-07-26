"""Backtest Kronos on intraday (60-minute) candles: predict every hourly candle of
a trading day from the preceding N trading days of hourly history, walk-forward
across several trading days.

Unlike nse/forecast.py (which always predicts a single end-of-day close), this
predicts the *whole* next day's hourly path at once -- pred_len equals however
many intraday bars that day actually has.

Example (5 trading days of context, predicting each of the last 10 trading days
before HDFC Bank's Q1 FY27 results, so the earnings shock doesn't contaminate
the test):
  python -m nse.intraday_backtest --csv data/NSE_HDFCBANK_60min_range.csv \\
      --lookback-days 5 --num-days 10 --before 2026-07-18 \\
      --output data/NSE_HDFCBANK_intraday_backtest.csv \\
      --chart-output data/NSE_HDFCBANK_intraday_backtest.png
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


def load_intraday(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    return df.sort_values("timestamps").reset_index(drop=True)


def trading_dates(df: pd.DataFrame):
    return sorted(df["timestamps"].dt.normalize().unique())


def backtest_day(df: pd.DataFrame, target_date, lookback_days: int, predictor, T, top_p, sample_count):
    all_dates = trading_dates(df)
    prior_dates = [d for d in all_dates if d < target_date][-lookback_days:]
    if len(prior_dates) < lookback_days:
        return None
    context = df[df["timestamps"].dt.normalize().isin(prior_dates)]
    target_bars = df[df["timestamps"].dt.normalize() == target_date].reset_index(drop=True)
    if target_bars.empty:
        return None
    pred_len = len(target_bars)
    y_timestamp = target_bars["timestamps"]
    pred = predictor.predict(
        df=context[PRICE_COLS],
        x_timestamp=context["timestamps"],
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=T,
        top_p=top_p,
        sample_count=sample_count,
        verbose=False,
    )
    pred.index = y_timestamp.values
    actual = target_bars.set_index("timestamps")[PRICE_COLS]
    return context, actual, pred


def compute_metrics(actual, pred, column="close"):
    errors = pred[column].to_numpy() - actual[column].to_numpy()
    mae = abs(errors).mean()
    rmse = (errors**2).mean() ** 0.5
    mape = (abs(errors) / actual[column].to_numpy()).mean() * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE_pct": mape}


def plot_days(results, symbol, chart_output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.2 * rows), squeeze=False)

    for i, (target_date, actual, pred, metrics) in enumerate(results):
        ax = axes[i // cols][i % cols]
        hours = [t.strftime("%H:%M") for t in actual.index]
        ax.plot(hours, actual["close"].values, color="#2ca02c", marker="o", markersize=4, label="Actual")
        ax.plot(hours, pred["close"].values, color="#9467bd", marker="^", markersize=4, linestyle="--", label="Predicted")
        ax.set_title(f"{target_date.date()}\nMAE={metrics['MAE']:.2f}", fontsize=9)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle(f"{symbol} — intraday (hourly) backtest, {n} days x {len(results[0][1])} bars/day", fontsize=12)
    fig.tight_layout()
    fig.savefig(chart_output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, help="path to an intraday OHLCV CSV (e.g. 60minute candles)")
    parser.add_argument("--lookback-days", type=int, default=5, help="trading days of intraday history to use as context (default 5)")
    parser.add_argument("--num-days", type=int, default=10, help="number of trading days to walk-forward test (default 10)")
    parser.add_argument("--before", help="only test trading days strictly before this date (YYYY-MM-DD), e.g. to stay clear of an earnings event")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--output", help="path to write the per-bar backtest comparison CSV to")
    parser.add_argument("--chart-output", help="path to write a per-day small-multiples PNG chart to")
    args = parser.parse_args()

    df = load_intraday(args.csv)
    symbol = Path(args.csv).stem

    all_dates = trading_dates(df)
    if args.before:
        all_dates = [d for d in all_dates if d < pd.Timestamp(args.before)]
    if len(all_dates) <= args.lookback_days:
        raise SystemExit(f"Not enough trading days before the cutoff (need > {args.lookback_days}, got {len(all_dates)})")
    target_dates = all_dates[-args.num_days:]

    print(f"Loading {MODEL_NAME} / {TOKENIZER_NAME} ...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
    model = Kronos.from_pretrained(MODEL_NAME)
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    print(f"Backtesting {len(target_dates)} trading days, each predicted from the {args.lookback_days} trading days before it ...")
    results = []
    all_comparisons = []
    for target_date in target_dates:
        outcome = backtest_day(df, target_date, args.lookback_days, predictor, args.temperature, args.top_p, args.sample_count)
        if outcome is None:
            continue
        context, actual, pred = outcome
        metrics = compute_metrics(actual, pred, column="close")
        results.append((target_date, actual, pred, metrics))
        print(
            f"  {target_date.date()}: {len(actual)} bars, "
            f"MAE={metrics['MAE']:.2f} RMSE={metrics['RMSE']:.2f} MAPE={metrics['MAPE_pct']:.2f}%"
        )
        comparison = pd.DataFrame({
            "actual_close": actual["close"].values,
            "predicted_close": pred["close"].values,
            "error": pred["close"].values - actual["close"].values,
        }, index=actual.index)
        all_comparisons.append(comparison)

    combined = pd.concat(all_comparisons)
    overall = compute_metrics(
        combined.rename(columns={"actual_close": "close"}),
        combined.rename(columns={"predicted_close": "close"}),
        column="close",
    )
    print(
        f"\nOverall across {len(results)} days / {len(combined)} hourly bars: "
        f"MAE={overall['MAE']:.2f}  RMSE={overall['RMSE']:.2f}  MAPE={overall['MAPE_pct']:.2f}%"
    )

    if args.output:
        combined.to_csv(args.output)
        print(f"Wrote per-bar comparison to {args.output}")

    if args.chart_output:
        plot_days(results, symbol, args.chart_output)
        print(f"Wrote chart to {args.chart_output}")


if __name__ == "__main__":
    main()
