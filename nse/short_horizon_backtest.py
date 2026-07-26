"""How low can the context/horizon go? Rolling short-horizon test on 5-minute
candles: feed --context-hours of 5-minute bars, predict --horizon-minutes ahead,
sliding the window forward by --step-minutes through every trading day (no
cross-day context) to build up a real sample size instead of one shot per day.

Example (1 hour of context -> 10 minutes ahead, stepped every 10 minutes,
across the 10 trading days before HDFC Bank's Q1 FY27 results):
  python -m nse.short_horizon_backtest --csv data/NSE_HDFCBANK_5min_range.csv \\
      --context-hours 1 --horizon-minutes 10 --step-minutes 10 --before 2026-07-18 \\
      --output data/NSE_HDFCBANK_short_horizon.csv \\
      --chart-output data/NSE_HDFCBANK_short_horizon.png
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
BAR_MINUTES = 5


def load_5min(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    return df.sort_values("timestamps").reset_index(drop=True)


def rolling_short_horizon(df, context_bars, horizon_bars, step_bars, predictor, T, top_p, sample_count):
    rows = []
    for date, day_bars in df.groupby(df["timestamps"].dt.normalize()):
        day_bars = day_bars.reset_index(drop=True)
        start = context_bars
        while start + horizon_bars <= len(day_bars):
            context = day_bars.iloc[start - context_bars: start]
            target = day_bars.iloc[start: start + horizon_bars]
            y_timestamp = target["timestamps"]
            pred = predictor.predict(
                df=context[PRICE_COLS],
                x_timestamp=context["timestamps"],
                y_timestamp=y_timestamp,
                pred_len=horizon_bars,
                T=T,
                top_p=top_p,
                sample_count=sample_count,
                verbose=False,
            )
            rows.append({
                "date": date.date(),
                "window_start": context["timestamps"].iloc[0].strftime("%H:%M"),
                "predict_from": context["timestamps"].iloc[-1].strftime("%H:%M"),
                "predict_to": target["timestamps"].iloc[-1].strftime("%H:%M"),
                "given_end_price": context["close"].iloc[-1],
                "actual_close": target["close"].iloc[-1],
                "predicted_close": pred["close"].iloc[-1],
            })
            start += step_bars
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    errors = df["predicted_close"] - df["actual_close"]
    mae = errors.abs().mean()
    rmse = (errors**2).mean() ** 0.5
    mape = (errors.abs() / df["actual_close"]).mean() * 100
    direction_correct = (
        (df["predicted_close"] > df["given_end_price"]) == (df["actual_close"] > df["given_end_price"])
    ).mean() * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE_pct": mape, "direction_pct": direction_correct}


def plot_summary(df: pd.DataFrame, symbol, context_hours, horizon_minutes, chart_output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = df.copy()
    df["error"] = df["predicted_close"] - df["actual_close"]
    df["abs_pct_error"] = (df["error"].abs() / df["actual_close"]) * 100

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].scatter(df["actual_close"], df["predicted_close"], alpha=0.4, s=15, color="#1f77b4")
    lims = [min(df["actual_close"].min(), df["predicted_close"].min()), max(df["actual_close"].max(), df["predicted_close"].max())]
    axes[0].plot(lims, lims, color="gray", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Actual close")
    axes[0].set_ylabel("Predicted close")
    axes[0].set_title("Predicted vs actual")

    axes[1].hist(df["error"], bins=30, color="#9467bd", edgecolor="white")
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_xlabel("Error (predicted - actual)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Error distribution (n={len(df)})")

    by_time = df.groupby("predict_from")["abs_pct_error"].mean().sort_index()
    axes[2].plot(by_time.index, by_time.values, marker="o", markersize=3, color="#d62728")
    axes[2].tick_params(axis="x", labelrotation=90, labelsize=6)
    axes[2].set_ylabel("Mean |% error|")
    axes[2].set_title("Error by time of day")

    fig.suptitle(f"{symbol} — {context_hours}hr context -> {horizon_minutes}min ahead, rolling through each day (same-day only)", fontsize=12)
    fig.tight_layout()
    fig.savefig(chart_output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, help="path to a 5-minute OHLCV CSV")
    parser.add_argument("--context-hours", type=float, default=1, help="hours of 5-minute candles to feed in (default 1)")
    parser.add_argument("--horizon-minutes", type=int, default=10, help="minutes ahead to predict (default 10)")
    parser.add_argument("--step-minutes", type=int, default=10, help="how far to slide the window forward each step (default 10, i.e. non-overlapping)")
    parser.add_argument("--before", help="only test trading days strictly before this date (YYYY-MM-DD)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--output", help="path to write the per-window comparison CSV to")
    parser.add_argument("--chart-output", help="path to write a summary PNG chart to")
    args = parser.parse_args()

    df = load_5min(args.csv)
    symbol = Path(args.csv).stem
    if args.before:
        df = df[df["timestamps"] < pd.Timestamp(args.before)]

    context_bars = int(round(args.context_hours * 60 / BAR_MINUTES))
    horizon_bars = max(1, round(args.horizon_minutes / BAR_MINUTES))
    step_bars = max(1, round(args.step_minutes / BAR_MINUTES))

    print(f"Loading {MODEL_NAME} / {TOKENIZER_NAME} ...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
    model = Kronos.from_pretrained(MODEL_NAME)
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    print(
        f"Rolling {args.context_hours}hr context ({context_bars} bars) -> {args.horizon_minutes}min ahead "
        f"({horizon_bars} bars), step {args.step_minutes}min ({step_bars} bars), same-day only ..."
    )
    result = rolling_short_horizon(df, context_bars, horizon_bars, step_bars, predictor, args.temperature, args.top_p, args.sample_count)

    metrics = summarize(result)
    print(f"\n{len(result)} windows across {result['date'].nunique()} days")
    print(
        f"MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}  MAPE={metrics['MAPE_pct']:.2f}%  "
        f"direction correct={metrics['direction_pct']:.1f}%"
    )

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"Wrote comparison to {args.output}")
    if args.chart_output:
        plot_summary(result, symbol, args.context_hours, args.horizon_minutes, args.chart_output)
        print(f"Wrote chart to {args.chart_output}")


if __name__ == "__main__":
    main()
