"""Same-day-only test: feed a trading day's own first N hours of 5-minute candles
(no other days' history at all) and predict that day's final close as a single
value, walked across several trading days.

Example (first 4 hours of 5-minute candles -> predict the close, across the 10
trading days before HDFC Bank's Q1 FY27 results):
  python -m nse.intraday_5min_close --csv data/NSE_HDFCBANK_5min_range.csv \\
      --hours 4 --before 2026-07-18 \\
      --output data/NSE_HDFCBANK_5min_close.csv \\
      --chart-output data/NSE_HDFCBANK_5min_close.png
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
BARS_PER_HOUR_5MIN = 12


def load_5min(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    return df.sort_values("timestamps").reset_index(drop=True)


def trading_dates(df: pd.DataFrame):
    return sorted(df["timestamps"].dt.normalize().unique())


def predict_close_from_hours(df: pd.DataFrame, target_date, hours: int, predictor, T, top_p, sample_count):
    day_bars = df[df["timestamps"].dt.normalize() == target_date].reset_index(drop=True)
    context_bars = int(round(hours * BARS_PER_HOUR_5MIN))
    if len(day_bars) <= context_bars:
        return None

    context = day_bars.iloc[:context_bars]
    final_bar = day_bars.iloc[-1]
    y_timestamp = pd.Series([final_bar["timestamps"]])
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
    predicted_close = pred["close"].iloc[0]
    return context, final_bar, predicted_close


def plot_results(results, symbol, hours, chart_output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.2 * rows), squeeze=False)

    for i, (target_date, context, final_bar, predicted_close, error) in enumerate(results):
        ax = axes[i // cols][i % cols]
        times = [t.strftime("%H:%M") for t in context["timestamps"]]
        ax.plot(times, context["close"], color="#1f77b4", linewidth=1.2, label=f"First {hours}hrs (given)")
        ax.axhline(final_bar["close"], color="#2ca02c", linestyle="-", linewidth=1.5, label="Actual close")
        ax.axhline(predicted_close, color="#9467bd", linestyle="--", linewidth=1.5, label="Predicted close")
        ax.set_title(f"{target_date.date()}\nerror={error:+.2f}", fontsize=9)
        ax.set_xticks(times[::12])
        ax.tick_params(axis="x", labelrotation=45, labelsize=6.5)
        ax.tick_params(axis="y", labelsize=7)
        if i == 0:
            ax.legend(fontsize=6.5)

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle(f"{symbol} — predict close from first {hours}hrs of 5min candles, {n} days (same-day only, no cross-day context)", fontsize=11)
    fig.tight_layout()
    fig.savefig(chart_output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, help="path to a 5-minute OHLCV CSV")
    parser.add_argument("--hours", type=float, default=4, help="hours of 5-minute candles to feed in per day (default 4)")
    parser.add_argument("--num-days", type=int, help="number of most recent trading days to test (default: all in --csv)")
    parser.add_argument("--before", help="only test trading days strictly before this date (YYYY-MM-DD)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--output", help="path to write the per-day comparison CSV to")
    parser.add_argument("--chart-output", help="path to write a per-day small-multiples PNG chart to")
    args = parser.parse_args()

    df = load_5min(args.csv)
    symbol = Path(args.csv).stem

    dates = trading_dates(df)
    if args.before:
        dates = [d for d in dates if d < pd.Timestamp(args.before)]
    if args.num_days:
        dates = dates[-args.num_days:]

    print(f"Loading {MODEL_NAME} / {TOKENIZER_NAME} ...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
    model = Kronos.from_pretrained(MODEL_NAME)
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    print(f"Predicting the close from the first {args.hours} hours of 5-minute candles, {len(dates)} days, same-day only ...")
    results = []
    rows = []
    for target_date in dates:
        outcome = predict_close_from_hours(df, target_date, args.hours, predictor, args.temperature, args.top_p, args.sample_count)
        if outcome is None:
            continue
        context, final_bar, predicted_close = outcome
        error = predicted_close - final_bar["close"]
        results.append((target_date, context, final_bar, predicted_close, error))
        print(
            f"  {target_date.date()}: given close (end of {args.hours}hrs)={context['close'].iloc[-1]:.2f}, "
            f"actual close={final_bar['close']:.2f}, predicted close={predicted_close:.2f}, error={error:+.2f}"
        )
        rows.append({
            "date": target_date.date(),
            "given_end_price": context["close"].iloc[-1],
            "actual_close": final_bar["close"],
            "predicted_close": predicted_close,
            "error": error,
        })

    result_df = pd.DataFrame(rows)
    mae = result_df["error"].abs().mean()
    rmse = (result_df["error"] ** 2).mean() ** 0.5
    mape = (result_df["error"].abs() / result_df["actual_close"]).mean() * 100
    print(f"\nOverall across {len(result_df)} days: MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%")

    if args.output:
        result_df.to_csv(args.output, index=False)
        print(f"Wrote comparison to {args.output}")
    if args.chart_output:
        plot_results(results, symbol, args.hours, args.chart_output)
        print(f"Wrote chart to {args.chart_output}")


if __name__ == "__main__":
    main()
