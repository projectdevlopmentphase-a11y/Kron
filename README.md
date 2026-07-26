# Kron

NSE equity forecasting on top of [Kronos](https://github.com/shiyu-coder/Kronos), an
open-source foundation model for financial candlesticks (K-lines).

## Layout

- `kronos/model/` — vendored Kronos model code (MIT licensed, from shiyu-coder/Kronos).
- `nse/kite_client.py` — pulls NSE equity OHLCV candles via Kite Connect (`kiteconnect`).
- `nse/forecast.py` — CLI that loads `Kronos-small` + `Kronos-Tokenizer-base` from
  Hugging Face and forecasts future candles from historical ones.
- `nse/intraday_backtest.py` — CLI that backtests Kronos on intraday (hourly)
  candles: predicts a whole trading day's hourly path from the N trading days
  before it, walk-forward across several days.
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

`--append-csv`/`--append-tail` also work with `--backtest`: at every
walk-forward step, the prior day's single daily bar is swapped out for that
same day's last `--append-tail` intraday candles (matched by date):

```shell
python -m nse.forecast --csv data/NSE_HDFCBANK_day.csv --lookback 400 --pred-len 20 \
    --backtest --append-csv data/NSE_HDFCBANK_60min_range.csv --append-tail 2 \
    --event "2026-07-18:Q1 FY27 results" \
    --output data/NSE_HDFCBANK_backtest_intraday.csv \
    --chart-output data/NSE_HDFCBANK_backtest_intraday.png
```

On the HDFCBANK earnings-week window this made things *worse* (MAE 16.08 vs
11.45 for the daily-only walk-forward, see
`data/NSE_HDFCBANK_backtest_daily_vs_intraday.png`) — folding in 2 hours of
intraday swings as the most recent context made the day-ahead forecast
noisier rather than more accurate, most visibly on 2026-07-21 where it
overshot the actual close by ~₹49. Splicing a couple of hours of a
different, finer granularity onto the end of an otherwise-daily series
seems to read to Kronos as a volatility-regime change rather than "more
information."

`--intraday-only` goes further: it drops the daily lookback entirely and
builds the *whole* context for every walk-forward step from
`--intraday-lookback` intraday candles (default 400, ~57 trading days),
instead of mixing granularities:

```shell
python -m nse.forecast --csv data/NSE_HDFCBANK_day.csv --lookback 400 --pred-len 20 \
    --backtest --append-csv data/NSE_HDFCBANK_60min_range.csv \
    --intraday-only --intraday-lookback 400 \
    --event "2026-07-18:Q1 FY27 results" \
    --output data/NSE_HDFCBANK_backtest_intraday_only.csv \
    --chart-output data/NSE_HDFCBANK_backtest_intraday_only.png
```

This is the best of the three (see `data/NSE_HDFCBANK_backtest_three_way.png`):

| Context | MAE | RMSE | MAPE |
|---|---|---|---|
| Daily-only walk-forward | 11.45 | 14.88 | 1.44% |
| Daily + last 2hrs intraday spliced on | 16.08 | 19.82 | 2.02% |
| Full 400-bar intraday context (no daily mixing) | **8.79** | **12.22** | **1.10%** |

So mixing granularities hurt, but going all-in on a *consistent* intraday
granularity for the whole context helped — including tracking the
earnings-week drop noticeably better than the daily-only version. This is
still one earnings week on one symbol, not a validated result, but it
suggests Kronos wants a single consistent candle interval throughout its
context rather than a blend.

More lookback isn't automatically better, though. Doubling to
`--intraday-lookback 800` (~8 months of hourly candles, back to
2025-12-01) made things drastically *worse*:

| Context | MAE | RMSE | MAPE |
|---|---|---|---|
| 400-bar intraday context (~57 trading days) | **8.79** | **12.22** | **1.10%** |
| 800-bar intraday context (~114 trading days) | 72.00 | 83.49 | 9.07% |

See `data/NSE_HDFCBANK_backtest_400_vs_800.png`. The 800-bar window reaches
back to 2026-01-06, when HDFCBANK was still trading around ₹950-960 —
noticeably higher than the ~₹795-800 it had settled to by late June. That
window's mean close is ~₹832, pulled up by the stale January price level.
`KronosPredictor.predict` normalizes its input by the mean/std of the
*whole* context, so that older, higher-priced regime biased every forecast
upward for most of the backtest (predictions consistently 50-150 points
above actual). The lesson: lookback should span enough history to be
informative, but not so much that it drags in a price regime the stock has
since moved away from — more context only helps as long as it's still
representative of current price level and volatility.

## Intraday backtesting

`nse/intraday_backtest.py` is a separate mode: instead of predicting a single
end-of-day close, it predicts every hourly candle of a trading day at once
(`pred_len` = however many bars that day has) from the `--lookback-days`
trading days immediately before it, walked forward across `--num-days`
trading days. Use `--before` to keep the whole test clear of a known event
like an earnings release:

```shell
python -m nse.intraday_backtest --csv data/NSE_HDFCBANK_60min_range.csv \
    --lookback-days 5 --num-days 10 --before 2026-07-18 \
    --output data/NSE_HDFCBANK_intraday_backtest.csv \
    --chart-output data/NSE_HDFCBANK_intraday_backtest.png
```

Backtesting the 10 trading days from 2026-07-06 to 2026-07-17 (all before the
Q1 FY27 results, each predicted from the 5 trading days before it) gives
**MAE 11.13, RMSE 14.03, MAPE 1.35%** across 70 hourly bars in this default
"single-shot" mode — see `data/NSE_HDFCBANK_intraday_backtest.png` for the
per-day small multiples. Most days track closely (2026-07-10 MAE 1.44,
2026-07-15 MAE 4.03), with the worst day (2026-07-06, MAE 28.90) being a
gap-up Monday: HDFCBANK rallied from Friday's ~₹801 close to open around
₹805 and race to ₹821-830 within the first hour, something the 5 prior
(calm, ₹795-805-range) trading days gave no hint of.

Single-shot mode generates all 7 hours of a day from one `predict()` call —
Kronos's own autoregressive generation chains hour 1's *guess* into hour 2,
hour 3, etc., but it never sees that day's *real* bars as they land, so once
the first guess misses a surprise like a gap-up, every later hour just keeps
extrapolating the same wrong guess instead of correcting to reality.
`--walk-forward` fixes exactly that: it predicts one bar at a time, feeding
each bar's real outcome back in as context before predicting the next one
(like `forecast.py`'s daily `--backtest`, but within the day):

```shell
python -m nse.intraday_backtest --csv data/NSE_HDFCBANK_60min_range.csv \
    --lookback-days 5 --num-days 10 --before 2026-07-18 --walk-forward \
    --output data/NSE_HDFCBANK_intraday_backtest_wf.csv \
    --chart-output data/NSE_HDFCBANK_intraday_backtest_wf.png
```

This drops overall MAE from 11.13 to **3.32** (RMSE 4.93, MAPE 0.40%) across
the same 70 bars. The gap-up day improves from MAE 28.90 to 6.31 — see
`data/NSE_HDFCBANK_intraday_0706_before_after.png`: the 09:15 bar still
misses (nothing before market open could know about the gap), but the
moment the real 09:15 close is fed back in, the 10:15 prediction jumps from
~₹798 to ~₹815 and tracks closely for the rest of the day. The lesson mirrors
the daily case: Kronos can't predict a surprise before it happens, but it
adapts to new real information almost immediately once it's given that
information, rather than compounding its own earlier miss.

### Live intraday use

`--backtest`/`--walk-forward` only work retroactively — they need a target
day's real outcome already in the CSV to score against. For live use, pass
`--predict-next N` instead: it takes whatever real candles are already in
your CSV, uses the last `--lookback-days` trading days as context, and
predicts the next `N` hourly bars from there — no held-out actuals needed.
It correctly continues the *same* trading day if the CSV's last bar isn't
15:15 yet (a session in progress), or rolls to 09:15 the next business day
if it is:

```shell
python -m nse.intraday_backtest --csv data/NSE_HDFCBANK_60min_range.csv \
    --lookback-days 5 --predict-next 1 \
    --output data/NSE_HDFCBANK_next_hour.csv \
    --chart-output data/NSE_HDFCBANK_next_hour.png
```

The practical live workflow, matching how `--walk-forward` behaves in the
backtest: pull fresh 60-minute candles (e.g. via `nse/kite_client.py` or a
live Kite session), append the newly-closed real bar to your CSV, then
re-run `--predict-next 1` to get a forecast for the *next* hour only. Repeat
after every hour closes throughout the session — each run picks up the real
bar you just appended as part of its context, which is exactly the
feed-the-real-bar-back-in mechanism `--walk-forward` simulates in
backtesting, just done live one hour at a time instead of replayed over
history. Asking for `--predict-next` with N > 1 in one shot reverts to the
single-shot behavior (no correction until you actually re-run it with new
data), so for live trading prefer N=1 and re-run every hour.

## Notes

Model weights (`NeoQuasar/Kronos-small`, `NeoQuasar/Kronos-Tokenizer-base`) are
downloaded from the Hugging Face Hub on first run and cached locally by
`huggingface_hub`. `huggingface.co` must be reachable for this step.
