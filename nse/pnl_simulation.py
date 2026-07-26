"""Simulate a simple daily long/short strategy on top of an intraday_backtest.py
output CSV: ₹--capital fresh each day (no compounding, no charges/slippage/STT),
direction decided from the model's forecast for that day's close made right after
the real 09:15 candle, position held until the actual close.

Example:
  python -m nse.pnl_simulation --intraday-csv data/NSE_HDFCBANK_60min_range.csv \\
      --backtest-csv data/NSE_HDFCBANK_intraday_backtest.csv --capital 10000 \\
      --output data/NSE_HDFCBANK_pnl_simulation.csv \\
      --chart-output data/NSE_HDFCBANK_pnl_simulation.png
"""
import argparse
from pathlib import Path

import pandas as pd


def simulate(intraday_csv: str, backtest_csv: str, capital: float) -> pd.DataFrame:
    intraday = pd.read_csv(intraday_csv, parse_dates=["timestamps"])
    backtest = pd.read_csv(backtest_csv, index_col=0, parse_dates=True)

    rows = []
    for day, group in backtest.groupby(backtest.index.normalize()):
        entry_bar = intraday[intraday["timestamps"] == day + pd.Timedelta(hours=9, minutes=15)]
        if entry_bar.empty:
            continue
        entry_price = entry_bar["close"].iloc[0]

        predicted_final_close = group["predicted_close"].iloc[-1]
        actual_final_close = group["actual_close"].iloc[-1]

        direction = "LONG" if predicted_final_close > entry_price else "SHORT"
        pct_move = (actual_final_close - entry_price) / entry_price
        pnl_pct = pct_move if direction == "LONG" else -pct_move

        rows.append({
            "date": day.date(),
            "entry_price": entry_price,
            "predicted_close": predicted_final_close,
            "actual_close": actual_final_close,
            "direction": direction,
            "pnl_pct": pnl_pct * 100,
            "pnl_rupees": capital * pnl_pct,
        })

    result = pd.DataFrame(rows)
    result["capital_end_of_day"] = capital + result["pnl_rupees"]
    return result


def plot_pnl(result: pd.DataFrame, capital: float, chart_output: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = result.copy()
    result["cumulative"] = result["pnl_rupees"].cumsum()
    labels = pd.to_datetime(result["date"]).dt.strftime("%m-%d")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})

    colors = ["#2ca02c" if p > 0 else "#d62728" for p in result["pnl_rupees"]]
    ax1.bar(labels, result["pnl_rupees"], color=colors)
    for x, (p, d) in enumerate(zip(result["pnl_rupees"], result["direction"])):
        ax1.annotate(f"{p:+.0f}\n({d})", (x, p), textcoords="offset points", xytext=(0, 5 if p >= 0 else -22), ha="center", fontsize=8)
    ax1.axhline(0, color="black", linewidth=0.8)
    pad = max(20, result["pnl_rupees"].abs().max() * 0.2)
    ax1.set_ylim(result["pnl_rupees"].min() - pad, result["pnl_rupees"].max() + pad)
    ax1.set_ylabel("Daily P&L (₹)")
    ax1.set_title("long/short decided from model's forecast right after the open, exit at actual close", fontsize=9.5)
    ax1.grid(axis="y", alpha=0.3)

    fig.suptitle(f"simulated daily strategy, ₹{capital:,.0f} fresh each day, no charges", fontsize=13, y=0.99)

    ax2.plot(labels, result["cumulative"], marker="o", color="#1f77b4")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Cumulative P&L (₹)")
    ax2.set_xlabel("Date")
    ax2.grid(axis="y", alpha=0.3)
    for x, c in enumerate(result["cumulative"]):
        ax2.annotate(f"{c:+.0f}", (x, c), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(chart_output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--intraday-csv", required=True, help="path to the raw intraday OHLCV CSV")
    parser.add_argument("--backtest-csv", required=True, help="path to an intraday_backtest.py --output CSV")
    parser.add_argument("--capital", type=float, default=10000, help="capital deployed fresh each day (default 10000)")
    parser.add_argument("--output", help="path to write the per-day P&L CSV to")
    parser.add_argument("--chart-output", help="path to write the P&L chart to")
    args = parser.parse_args()

    result = simulate(args.intraday_csv, args.backtest_csv, args.capital)
    print(result.to_string(index=False))
    print(f"\nTotal P&L over {len(result)} days (₹{args.capital:,.0f} fresh each day, no charges): ₹{result['pnl_rupees'].sum():.2f}")
    print(f"Winning days: {(result['pnl_rupees'] > 0).sum()} / {len(result)}")
    print(f"Average daily P&L: ₹{result['pnl_rupees'].mean():.2f} ({result['pnl_pct'].mean():.3f}%)")

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"\nWrote P&L to {args.output}")
    if args.chart_output:
        plot_pnl(result, args.capital, args.chart_output)
        print(f"Wrote chart to {args.chart_output}")


if __name__ == "__main__":
    main()
