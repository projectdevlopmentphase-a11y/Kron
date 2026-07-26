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
- `nse/intraday_5min_close.py` — CLI that predicts a day's final close from
  just that day's own first N hours of 5-minute candles (no cross-day context).
- `nse/short_horizon_backtest.py` — CLI that rolls a small context window
  through every trading day predicting a few minutes ahead each time, to
  probe how little context Kronos can work with.
- `nse/pnl_simulation.py` — turns an `intraday_backtest.py` output into a toy
  long/short daily P&L simulation.
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

`--given-bars N` generalizes "give the first candle" to any number of a
day's own opening candles — e.g. `--given-bars 3` feeds in 09:15-11:15 as
known context and only predicts/scores 12:15 onward:

```shell
python -m nse.intraday_backtest --csv data/NSE_HDFCBANK_60min_range.csv \
    --lookback-days 5 --num-days 10 --before 2026-07-18 --given-bars 3 \
    --output data/NSE_HDFCBANK_intraday_backtest_g3.csv \
    --chart-output data/NSE_HDFCBANK_intraday_backtest_g3.png
```

With 3 bars given, single-shot MAE improves to 4.63 and walk-forward to
2.84 (both down from 5.82 / 3.02 with just 1 bar given) — unsurprising,
since predicting only the last 4 hours of a day from 3 known real hours is
an easier task than predicting 6 hours from 1. More given context generally
makes the *remaining-bar* forecast easier, but it also means less of the
day is left to actually trade on.

### P&L simulation (no charges)

`nse/pnl_simulation.py` turns an `intraday_backtest.py` output CSV into a toy
daily long/short strategy: ₹`--capital` fresh each day (no compounding
across days, no brokerage/STT/slippage), direction decided from the model's
forecast for that day's close made right after the given opening bar(s),
exit at the actual close. Entry price/time is read from the backtest CSV
automatically, so it adapts to whatever `--given-bars` was used:

```shell
python -m nse.pnl_simulation --intraday-csv data/NSE_HDFCBANK_60min_range.csv \
    --backtest-csv data/NSE_HDFCBANK_intraday_backtest.csv --capital 10000 \
    --output data/NSE_HDFCBANK_pnl_simulation.csv \
    --chart-output data/NSE_HDFCBANK_pnl_simulation.png
```

Over the same 10 pre-earnings days with ₹10,000 deployed each day, entering
right after the 09:15 open (1 bar given): **net +₹97.42 total (+0.097%
average per day), but only 3 of 10 days were directionally correct** — see
`data/NSE_HDFCBANK_pnl_simulation.png`. The model called SHORT on most days
including several that actually rose (e.g. 2026-07-06, -₹85), but two large
correct SHORT calls on the two down days that followed (2026-07-07 +₹107,
2026-07-08 +₹255) covered the rest.

Re-running the same strategy off the `--given-bars 3` backtest (entering at
11:15 instead of 09:15, with 3 fewer hours left to capture any move) flips
the result to **net -₹355.88**, still with only 3/10 days directionally
correct — see `data/NSE_HDFCBANK_pnl_simulation_g3.png`. The lower MAE with
more bars given doesn't translate into a better trading outcome, because
MAE measures how close the *price level* forecast is, not whether the
*direction* call was right — and with a smaller remaining window there's
less room for a correct call to pay off before the day ends. Both results
are fragile outcomes from 10 days on one symbol, not a validated edge — a
slightly different window, or any real-world charge (brokerage, STT,
slippage), would likely change the sign of either one.

### 5-minute candles, same-day only

`nse/intraday_5min_close.py` is a different, more minimal experiment: no
cross-day context at all, no multi-day lookback -- just a trading day's own
first `--hours` of 5-minute candles (12 bars/hour) fed in, predicting that
same day's final close as a single value:

```shell
python -m nse.intraday_5min_close --csv data/NSE_HDFCBANK_5min_range.csv \
    --hours 4 --before 2026-07-18 \
    --output data/NSE_HDFCBANK_5min_close.csv \
    --chart-output data/NSE_HDFCBANK_5min_close.png
```

Feeding just the first 4 hours (48 bars) of each of the same 10 pre-earnings
days and predicting the 15:25 close gives **MAE 4.01, RMSE 6.41, MAPE
0.49%** — see `data/NSE_HDFCBANK_5min_close.png`. The predicted close
(purple dashed line) mostly hovers near wherever the first 4 hours left
off, which works well on most days since intraday closes don't usually
travel far from the late-morning level. The one outlier is 2026-07-08
(error +18.56): the stock kept sliding through the afternoon after a flat
morning, a continuation this same-day-only view had no way to anticipate
(no prior-day trend context to lean on, unlike the hourly `--lookback-days`
tests above). This is a much lighter-weight test than the multi-day hourly
backtests — no long history required, just today's own morning session —
and it holds up surprisingly well given how little it's given to work with.

`pnl_simulation.py --close-csv` runs the same toy strategy directly off an
`intraday_5min_close.py` output (self-contained, no separate raw-intraday
lookup needed): go long/short based on whether the predicted close is above
or below the price at the end of the given 4 hours (13:15), exit at the
actual close, ₹10,000 fresh each day, no charges:

```shell
python -m nse.pnl_simulation --close-csv data/NSE_HDFCBANK_5min_close.csv \
    --capital 10000 --output data/NSE_HDFCBANK_5min_pnl_simulation.csv \
    --chart-output data/NSE_HDFCBANK_5min_pnl_simulation.png
```

Result: **net -₹299.60** over the 10 days, 3/10 days directionally correct
— see `data/NSE_HDFCBANK_5min_pnl_simulation.png`. Same story as the hourly
version: a low MAE (4.01) doesn't imply a good hit rate, and the single big
miss (2026-07-08, predicted close ~828 vs actual 809.45) alone cost ₹218 of
the ₹300 net loss — a reminder that a single bad day can dominate a small
sample's P&L regardless of how accurate the model is on average.

### How low can the context/horizon go?

`nse/short_horizon_backtest.py` pushes this to the extreme: instead of one
prediction per day, it slides a small context window through *every*
trading day (same-day only, no cross-day history), predicting just a few
minutes ahead each time, to get a real sample size instead of 10 data
points:

```shell
python -m nse.short_horizon_backtest --csv data/NSE_HDFCBANK_5min_range.csv \
    --context-hours 1 --horizon-minutes 10 --step-minutes 10 --before 2026-07-18 \
    --output data/NSE_HDFCBANK_short_horizon.csv \
    --chart-output data/NSE_HDFCBANK_short_horizon.png
```

| Context | Horizon | Windows | MAE | MAPE | Direction correct |
|---|---|---|---|---|---|
| 1 hour | 10 min | 310 | 0.98 | 0.12% | 51.9% |
| 1 hour | 5 min | 630 | 0.72 | 0.09% | 51.4% |
| 30 min | 10 min | 340 | 0.99 | 0.12% | 46.2% |

Yes, you can feed in 1 hour and get a "reasonable" 10-minute prediction in
the sense that the price-level error is tiny (MAE well under ₹1, MAPE
~0.1%) — see `data/NSE_HDFCBANK_short_horizon.png` for the predicted-vs-actual
scatter (tight around the diagonal) and error distribution (narrow, centered
on zero). **But that low MAE is mostly a reflection of how small 10-minute
moves naturally are, not genuine predictive skill** — direction accuracy
across all three configs sits at 46-52%, indistinguishable from a coin
flip, and shrinking the context from 1 hour to 30 minutes made direction
accuracy *worse*, not better. So there's a floor here: below roughly an
hour of context, Kronos still produces plausible-looking numbers, but they
carry no more directional information than guessing.

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
