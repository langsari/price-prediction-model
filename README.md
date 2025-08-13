# price-prediction-model
A model to predict price movements and targets for stocks &amp; crypto. The algorithm identifies opportunities by analyzing volume, support, and resistance during classic breakout trading strategies.

# Price Prediction Model 
— Breakout Strategy (Stocks & Crypto)

## 1) Overview

A **price-prediction-model** that identifies trading opportunities using classic **breakout** logic with **support / resistance** and **volume confirmation**. The model predicts **near-term price movement** and **price targets** after a breakout for both **stocks** and **crypto**.

**Key goals**

* Detect breakout events reliably.
* Predict post‑breakout price direction and targets.
* Offer simple backtesting and evaluation.

> **Not financial advice. For research/education only.**

---

## 2) Features

* 📈 Support & resistance detection (rolling high/low)
* 🔔 Breakout detection with volume filter
* 🎯 Price target estimation using range projection
* 🤖 ML baseline (Random Forest / Linear Regression) with technical features
* 🔁 Simple backtest & evaluation (hit rate, RMSE, precision/recall for signals)
* 🧪 Jupyter notebooks for experimentation

---

## 3) Methodology (High‑level)

1. **Data**: OHLCV (Open, High, Low, Close, Volume) from Yahoo Finance (stocks) or crypto tickers.
2. **Indicators**:

   * SMA(20), SMA(50), RSI(14) *(customizable)*
   * Rolling **Support = min(Low, N)**, **Resistance = max(High, N)**
3. **Breakout rules** (default):

   * **Up-breakout** if `Close[t] > Resistance[t-1]` **and** `Volume[t] > MA_Volume(N)`
   * **Down-breakout** if `Close[t] < Support[t-1]` **and** `Volume[t] > MA_Volume(N)`
4. **Price targets**:

   * Define recent range: `Range_N = Resistance[t-1] - Support[t-1]`
   * **TargetUp = Close\[t] + k \* Range\_N**
   * \*\*TargetDown = Close\[t] - k \* Range\_N\`
   * Default `N = 20`, `k = 1.0` (configurable)
5. **Modeling**: Predict `Close[t+h]` (horizon `h`, e.g., 5 bars) &/or classification of **reach TargetUp/Down within H bars**.
6. **Evaluation**: RMSE/MAE for regression; precision/recall/F1 for breakout success; target‑hit rate.

---

## 4) Project Structure

```
price-prediction-model/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ data/                # (gitignored) raw/processed data
├─ notebooks/
│  ├─ 01_data_download.ipynb
│  ├─ 02_feature_engineering.ipynb
│  ├─ 03_breakout_detection.ipynb
│  ├─ 04_modeling_baseline.ipynb
│  └─ 05_backtest_evaluation.ipynb
├─ src/
│  ├─ config.py
│  ├─ data.py           # download & caching
│  ├─ features.py       # indicators, support/resistance
│  ├─ signals.py        # breakout logic & targets
│  ├─ models.py         # ML models & training
│  ├─ backtest.py       # simple backtest utilities
│  └─ utils.py
├─ docs/
│  └─ cover.png         # optional image for README
└─ .gitignore
```

---

## 5) Installation

```bash
# 1) Create env (optional)
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt
```

**requirements.txt** (example)

```
pandas>=2.0
numpy
matplotlib
scikit-learn
yfinance
TA-Lib; platform_system != 'Windows'
```

> *If TA‑Lib is hard to install, skip it—the repo uses only Pandas/Numpy by default.*

---

## 6) Quickstart

```bash
# Example: download BTC-USD daily data and run a quick experiment
python - <<'PY'
import pandas as pd, numpy as np, yfinance as yf
from datetime import date

TICKER = "BTC-USD"       # change to e.g. AAPL, ETH-USD
START  = "2023-01-01"
END    = str(date.today())

# 1) download
df = yf.download(TICKER, start=START, end=END).dropna()

# 2) features
N = 20
vol_ma = df['Volume'].rolling(N, min_periods=1).mean()
res = df['High'].rolling(N, min_periods=1).max()
sup = df['Low'].rolling(N, min_periods=1).min()

# 3) signals
break_up = (df['Close'] > res.shift(1)) & (df['Volume'] > vol_ma)
break_dn = (df['Close'] < sup.shift(1)) & (df['Volume'] > vol_ma)

# 4) targets (k=1.0)
rng = (res.shift(1) - sup.shift(1)).clip(lower=0)
TargetUp   = df['Close'] + rng
TargetDown = df['Close'] - rng

print("Signals summary:\n", pd.DataFrame({
    'Breakout_Up': break_up.sum(),
    'Breakout_Down': break_dn.sum()
}, index=[TICKER]))

print("\nSample rows:\n", pd.DataFrame({
    'Close': df['Close'].tail(5),
    'Res[-1]': res.shift(1).tail(5),
    'Sup[-1]': sup.shift(1).tail(5),
    'BreakUp': break_up.tail(5),
    'TargetUp': TargetUp.tail(5)
}))
PY
```

---

## 7) Data Sources

* **Yahoo Finance** via `yfinance` (OHLCV). Example tickers: `AAPL`, `TSLA` (stocks); `BTC-USD`, `ETH-USD` (crypto).
* **Timeframe**: configurable (daily/1h/15m if pulled via other APIs).

> Replace/add your real data sources if your instructor specifies.

---

## 8) Feature Engineering

* Moving Averages: `SMA20`, `SMA50`
* Momentum: `RSI(14)` *(optional)*
* Volatility: rolling high/low range `Range_N`
* Volume context: `Volume / MA_Volume(N)`

**Support/Resistance (rolling)**

```python
N = 20
Support    = Low.rolling(N, min_periods=1).min()
Resistance = High.rolling(N, min_periods=1).max()
```

---

## 9) Breakout Logic

```python
vol_ma = Volume.rolling(N, min_periods=1).mean()
Breakout_Up   = (Close > Resistance.shift(1)) & (Volume > vol_ma)
Breakout_Down = (Close < Support.shift(1))    & (Volume > vol_ma)
```

> You can tighten with a buffer, e.g. `Close > Resistance.shift(1) * 1.002` (0.2% filter).

---

## 10) Price Target Estimation

* **Projection by range** (default):

  * `Range_N = Resistance[t-1] - Support[t-1]`
  * `TargetUp   = Close[t] + k * Range_N`
  * `TargetDown = Close[t] - k * Range_N`
  * `k` in `[0.5, 1.5]` (tune via validation)
* **Alternative**: measure last consolidation width or ATR-based targets.

---

## 11) Modeling

Two task options (pick one or do both):

1. **Regression** — predict `Close[t+h]` (e.g., `h=5` bars)
2. **Classification** — predict whether **TargetUp/Down is hit within H bars**

**Baseline code (regression):**

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

features = pd.DataFrame({
    'SMA20': Close.rolling(20).mean(),
    'SMA50': Close.rolling(50).mean(),
    'Range20': (High.rolling(20).max() - Low.rolling(20).min()),
    'VolMA20': Volume.rolling(20).mean(),
    'VolRatio': Volume / Volume.rolling(20).mean()
}).dropna()

y = Close.shift(-5).reindex(features.index)  # predict 5 steps ahead
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, shuffle=False)

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print("RMSE:", rmse)
```

**Classification label example:**

```python
H = 10  # lookahead bars
hit_up = []
for i in range(len(Close)):
    fut = Close.iloc[i+1:i+1+H]
    tgt = (Close.iloc[i] + Range20.shift(1).iloc[i]) if i>0 else np.nan
    hit_up.append(bool((fut >= tgt).any()))
```

---

## 12) Backtest & Evaluation

**Metrics**

* Regression: RMSE, MAE, MAPE
* Classification: Precision/Recall/F1, ROC-AUC
* Strategy: Win rate, average R multiple, max drawdown *(optional)*

**Simple rule‑based backtest idea**

* Enter on `Breakout_Up` (or `Breakout_Down` for short)
* Exit when target hit or stop at opposite side of range
* Track PnL, drawdown

---

## 13) How to Run

```bash
# 1) Launch notebooks
jupyter lab  # or: jupyter notebook

# 2) Or run a quick script (example)
python src/example_run.py --ticker BTC-USD --start 2023-01-01 --end 2025-08-01 \
  --horizon 5 --range-window 20 --k 1.0
```

`src/example_run.py` (sketch):

```python
# pseudo: parse args -> download -> compute features -> detect breakout -> fit model -> report
```

---

## 14) Results (Examples)

Add your plots/tables here:

* `notebooks/04_modeling_baseline.ipynb` — predicted vs actual plot
* Target hit distribution
* Confusion matrix for breakout success

> Place sample images in `docs/` and reference them here.

---

## 15) Configuration

```python
# src/config.py
TICKER = "BTC-USD"
START  = "2023-01-01"
END    = "2025-08-01"
RANGE_WINDOW = 20
K_TARGET = 1.0
HORIZON = 5
```

---

## 16) Limitations

* Breakouts can fail in choppy/ranging markets.
* Using only OHLCV may miss fundamental/news context.
* Targets based on fixed range (`k * Range_N`) are simplistic.

---

## 17) Roadmap

* [ ] Add ATR‑based targets
* [ ] Add pattern filters (e.g., consolidation score)
* [ ] Hyperparameter search (Optuna)
* [ ] Position sizing & risk metrics
* [ ] Multi‑timeframe confirmation

---

## 18) Contributing

PRs and issues are welcome. Please open an issue for feature requests or bugs.

---

## 19) License

MIT — see [LICENSE](LICENSE).

---

## 20) Acknowledgements

* Yahoo Finance via `yfinance`
* scikit‑learn

---

