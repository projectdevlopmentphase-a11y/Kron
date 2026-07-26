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
end-of-day close, it predicts a trading day's hourly candles from the
`--lookback-days` trading days immediately before it, walked forward across
`--num-days` trading days. Use `--before` to keep the whole test clear of a
known event like an earnings release:

```shell
python -m nse.intraday_backtest --csv data/NSE_HDFCBANK_60min_range.csv \
    --lookback-days 5 --num-days 10 --before 2026-07-18 \
    --output data/NSE_HDFCBANK_intraday_backtest.csv \
    --chart-output data/NSE_HDFCBANK_intraday_backtest.png
```

Each day's own real **first candle (09:15) is always fed in as known context
and never predicted or scored** — it reflects the overnight/weekend gap,
which no amount of prior-day history can anticipate, so scoring it would
only measure something structurally unknowable. Only the remaining bars
(10:15 → 15:15, 6 of them) are generated.

Backtesting the 10 trading days from 2026-07-06 to 2026-07-17 (all before the
Q1 FY27 results, each predicted from the 5 trading days before it, plus that
day's own real open) gives **MAE 5.82, RMSE 7.28, MAPE 0.71%** across 60
hourly bars in this default "single-shot" mode — see
`data/NSE_HDFCBANK_intraday_backtest.png` for the per-day small multiples
(blue square marks the given first candle). The worst day is still
2026-07-06 (MAE 8.47), a gap-up Monday: HDFCBANK opened around ₹821 (already
given) and kept climbing to ₹828-830, a continuation the 5 calm prior days
gave no hint of.

Single-shot mode generates all 6 remaining hours from one `predict()` call —
Kronos's own autoregressive generation chains hour 2's *guess* into hour 3,
hour 4, etc., but it never sees that day's *real* bars as they land, so once
a guess drifts off, later hours just keep extrapolating instead of
correcting to reality. `--walk-forward` fixes exactly that: it predicts one
bar at a time, feeding each bar's real outcome back in as context before
predicting the next one (like `forecast.py`'s daily `--backtest`, but within
the day, and still starting from that day's given real open):

```shell
python -m nse.intraday_backtest --csv data/NSE_HDFCBANK_60min_range.csv \
    --lookback-days 5 --num-days 10 --before 2026-07-18 --walk-forward \
    --output data/NSE_HDFCBANK_intraday_backtest_wf.csv \
    --chart-output data/NSE_HDFCBANK_intraday_backtest_wf.png
```

This drops overall MAE from 5.82 to **3.02** (RMSE 4.14, MAPE 0.37%) across
the same 60 bars, and the gap-up day from 8.47 to 5.31 — see
`data/NSE_HDFCBANK_intraday_0706_before_after.png`: single-shot (red) stays
too low most of the day, while walk-forward (purple) climbs back toward
actual as each real hour gets fed back in, closing the gap almost entirely
by 15:15. The lesson mirrors the daily case: Kronos can't predict a surprise
before it happens, but once given the day's actual starting point, it
adapts to new real information quickly rather than compounding its own
earlier miss.

### P&L simulation (no charges)

`nse/pnl_simulation.py` turns an `intraday_backtest.py` output CSV into a toy
daily long/short strategy: ₹`--capital` fresh each day (no compounding
across days, no brokerage/STT/slippage), direction decided from the model's
forecast for that day's close made right after the real 09:15 candle, exit
at the actual close:

```shell
python -m nse.pnl_simulation --intraday-csv data/NSE_HDFCBANK_60min_range.csv \
    --backtest-csv data/NSE_HDFCBANK_intraday_backtest.csv --capital 10000 \
    --output data/NSE_HDFCBANK_pnl_simulation.csv \
    --chart-output data/NSE_HDFCBANK_pnl_simulation.png
```

Over the same 10 pre-earnings days with ₹10,000 deployed each day: **net
+₹97.42 total (+0.097% average per day), but only 3 of 10 days were
winners** — see `data/NSE_HDFCBANK_pnl_simulation.png`. The model called
SHORT on most days including several that actually rose (e.g. 2026-07-06,
-₹85), but two large correct SHORT calls on the two down days that followed
(2026-07-07 +₹107, 2026-07-08 +₹255) covered the rest. This is a fragile
result from 10 days on one symbol, not a validated edge — a slightly
different window or a couple of real-world charges (brokerage, STT,
slippage) would likely erase the ₹97 entirely. Directional single-shot
forecasts made once at the open, scored only on hitting long/short
correctly, are a much harder bar than the MAE numbers above suggest.

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
