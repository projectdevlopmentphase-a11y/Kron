# Kron

NSE equity forecasting on top of [Kronos](https://github.com/shiyu-coder/Kronos), an
open-source foundation model for financial candlesticks (K-lines).

## Layout

- `kronos/model/` — vendored Kronos model code (MIT licensed, from shiyu-coder/Kronos).
- `nse/kite_client.py` — pulls NSE equity OHLCV candles via Kite Connect (`kiteconnect`).
- `nse/forecast.py` — CLI that loads `Kronos-small` + `Kronos-Tokenizer-base` from
  Hugging Face and forecasts future candles from historical ones.
- `data/` — sample data and forecast output.

## Setup

```shell
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To pull live data, set Kite Connect credentials:

```shell
export KITE_API_KEY=...
export KITE_ACCESS_TOKEN=...
```

## Usage

Live data from Kite:

```shell
python -m nse.forecast --symbol HDFCBANK --from-date 2023-06-01 --to-date 2026-07-24 \
    --lookback 400 --pred-len 20 --output data/NSE_HDFCBANK_forecast.csv
```

Local CSV (columns: `timestamps, open, high, low, close, volume[, amount]`):

```shell
python -m nse.forecast --csv data/NSE_HDFCBANK_day.csv --lookback 400 --pred-len 20
```

Backtest against the most recent known candles: holds out the last `--pred-len`
candles and walk-forward scores Kronos against what actually happened — it
re-forecasts one day at a time, feeding each day's *actual* close back in as
context before predicting the next day (a "re-run every morning with last
night's real close" simulation) — reporting MAE/RMSE/MAPE on close. Optionally
annotate real-world events on the chart:

```shell
python -m nse.forecast --csv data/NSE_HDFCBANK_day.csv --lookback 400 --pred-len 20 \
    --backtest --event "2026-07-18:Q1 FY27 results" --event "2026-07-20:NIM miss reaction" \
    --output data/NSE_HDFCBANK_backtest.csv --chart-output data/NSE_HDFCBANK_backtest.png
```

`data/NSE_HDFCBANK_backtest.png` shows Kronos tracking most sessions
reasonably but lagging around HDFC Bank's Q1 FY27 results (announced
2026-07-18, market reaction 2026-07-20 on NIM compression and a profit/NII
miss vs street estimates) — error jumps right after the results and then
recovers over the following sessions as real closes get fed back in. Kronos
is a pure price-based model with no fundamentals/news input, so it can't
anticipate a discrete earnings surprise before it happens — it can only
adapt once the actual post-results prices are visible.

## Notes

Model weights (`NeoQuasar/Kronos-small`, `NeoQuasar/Kronos-Tokenizer-base`) are
downloaded from the Hugging Face Hub on first run and cached locally by
`huggingface_hub`. `huggingface.co` must be reachable for this step.
