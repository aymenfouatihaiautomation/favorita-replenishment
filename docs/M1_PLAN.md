# M1 plan: sales forecasting + replenishment policy (Favorita)

Business question: every Monday, how many units should each store order for each item,
to cover the next 9 days (7-day period + 2-day delivery delay)?
And how much stock-out + overstock cost does it save compared with "ordering as usual"?

Target: M1 finished around 10 December 2026, submitted to ML Zoomcamp Capstone 1 (deadline 5 January 2027).

## How to use this plan

- One step at a time, in order. Each step ends with a commit.
- "Who" says who writes the code:
  - **Aymen**: I write it myself. Claude Code may review it or write its tests, never rewrite it.
  - **Claude Code**: Claude Code writes it, Aymen reviews it before the commit.
- A step is done only when every line of "Done when" is true.
- Project rules are in CLAUDE.md.

## Step 0. Data preparation (DONE, commit c77d3ac)

- Who: Claude Code
- Output: `data/processed/sales.parquet` (10 stores x 200 items x 956 days = 1,912,000 rows),
  `holidays_events.parquet`, `oil.parquet`
- Done when: 7 data tests pass. ✅

## Step 1. Seasonal naive baseline ("ordering as usual")

- Who: **Aymen** (`src/baseline.py`); Claude Code writes `tests/test_baseline.py`
- What: for each row (store, item, date), the forecast is the mean of the sales of the same
  store and item 14, 21, 28 and 35 days earlier (same weekday, all known on the order date).
- Pitfall: use **dates**, not row positions (December 25 is missing from the grid).
- Done when:
  - `seasonal_naive(sales)` returns one forecast per row;
  - tests pass on a small hand-made example;
  - the first 35 days of each series have no forecast (not enough history), and this is expected.

## Step 2. Rolling backtest

- Who: **Aymen** (`src/backtest.py`); Claude Code writes the tests
- What:
  - 16 order dates: the last 16 Mondays before 2017-08-15;
  - for each order date, forecast the next 9 days using only data before the order date;
  - metrics: WAPE = sum(|error|) / sum(sales), and bias = sum(forecast - sales) / sum(sales);
  - results overall, per family, and perishable vs non-perishable.
- Done when:
  - the baseline is evaluated: one WAPE and one bias, overall and per family;
  - a test proves that no data on or after the order date is used for training.

## Step 3. Features

- Who: **Aymen** (`src/features.py`); Claude Code writes the tests
- Features (all known on the order date):
  - calendar: day of week, day of month, month, payday (15th and last day of month);
  - holidays: national, regional (store state), local (store city); `transferred = True`
    means NOT a day off; the `Transfer` row gives the real day off;
  - promotion: `onpromotion` of the target day (planned in advance);
  - history: sales 14, 21, 28, 35 days before; mean of the last 4 weeks, ending at day -14;
  - identifiers: store, item, family, store type, perishable (categorical).
  - Earthquake: rows from 2016-04-16 to 2016-05-31 are excluded from training.
- Forbidden: `transactions.csv`, same-day data, lags < 14 days, statistics computed on the full period.
- Done when: a test checks that every feature for a date only uses data at least 14 days older.

## Step 4. LightGBM model

- Who: Claude Code (`src/train.py`), Aymen reviews
- What: one LightGBM model, trained inside the backtest of step 2 (same 16 order dates),
  early stopping on the last 4 weeks before each order date, `random_state` fixed.
- Done when:
  - model WAPE and bias are compared with the baseline, overall and per family;
  - the model beats the baseline overall (if not, we stop and analyse why);
  - feature importance is saved.

## Step 5. Ordering simulation and cost avoided

- Who: **Aymen** (`src/simulate.py`); Claude Code writes the tests
- What:
  - weekly order = forecast for the 9 days + safety stock - stock on hand;
  - safety stock = k x standard deviation of backtest errors; k comes from the critical ratio p / (p + h);
  - non-perishable leftovers carry over to the next week; perishable leftovers are thrown away at the end of the week;
  - cost = p x units missing + h x units left over;
  - compare "as usual" (baseline forecast) and "model" (LightGBM forecast) with the same rule;
  - sensitivity: p / h from 2 to 20.
- Done when: one table and one chart give the cost avoided (in % and in units) for each p / h,
  and the calculation can be redone by hand on one item and one week.

## Step 6. API and Docker

- Who: Claude Code (`api/`, `Dockerfile`, tests), Aymen reviews
- What: FastAPI service with `/health`, `/forecast` (store, item, date -> 9 daily forecasts)
  and `/order` (store, item, date, p, h -> recommended order and "as usual" order).
  Uses the trained model saved in step 4. Runs in Docker. Deployed on Railway.
- Done when: the API answers in Docker on my PC and online, API tests pass.

## Step 7. Showcase

- Who: Claude Code for the page code; **Aymen** for all texts, the README and the video
- What:
  - project page on my portfolio: problem, result (cost avoided), live demo (store, item, date,
    p and h sliders), how it works, limits, technologies, about me, contact;
  - README written by me: question, method, results, limits, how to run;
  - 2-minute demo video.
- Note: never show raw Kaggle data, only forecasts and results.
- Done when: the page is online and calls the deployed API; README and video are done.

## Step 8. Check without AI + Zoomcamp

- Who: **Aymen**, without any AI
- What:
  - code a rolling backtest from memory;
  - explain one data leak I avoided;
  - redo the cost-avoided calculation by hand.
- Then submit to ML Zoomcamp Capstone 1 (deadline 5 January 2027) and do 3 peer reviews.
