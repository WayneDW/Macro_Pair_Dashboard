import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===== Parameters =====
WINDOW = 250  # rolling window size (e.g., 250 trading days ≈ 1 year)

PAIRS = [
    ('TQQQ', 'GLD'),      # risk-on vs safe-haven (leveraged growth sentiment)
    ('QQQ', 'GLD'),       # risk-on vs safe-haven
    ('QQQ', 'SPY'),       # growth vs value
    ('QQQ', 'BRK-B'),     # growth vs quality value
    ('NVDA', 'QQQ'),      # AI & innovation vs broad market
    ('SMH', 'QQQ'),       # semiconductors vs tech index
    ('USO', 'TLT'),       # real-economy vs bonds
    ('TIP', 'TLT'),       # inflation expectation vs nominal rate
    ('DX-Y.NYB', 'GLD')   # dollar vs gold (global risk flow)
]



START_DATE = "2016-01-01"
END_DATE = "2025-10-20"

# ===== Helper Function =====
def get_rolling_return(df, sym, window=100):
    df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
    df[f'RollingCumReturn_{sym}_{window}'] = np.exp(df['LogReturn'].rolling(window=window).sum()) - 1
    return df[[f'RollingCumReturn_{sym}_{window}']].dropna()

# ===== Download all tickers only once =====
all_tickers = sorted(list(set([x for pair in PAIRS for x in pair])))
raw_data = {sym: yf.download(sym, start=START_DATE, end=END_DATE) for sym in all_tickers}

# ===== Prepare figure =====
n_rows = 3
fig, axs = plt.subplots(n_rows, 3, figsize=(14, n_rows * 3.5))
axs = axs.flatten()

# ===== Loop over pairs =====
for idx, (TARGET, BASE) in enumerate(PAIRS):
    ax = axs[idx]
    target_raw = raw_data[TARGET].copy()
    base_raw = raw_data[BASE].copy()

    # Compute rolling returns
    q1 = get_rolling_return(target_raw, TARGET, window=WINDOW)
    q2 = get_rolling_return(base_raw, BASE, window=WINDOW)

    df = pd.merge(q1, q2, left_index=True, right_index=True, how='inner')

    diff = df[f'RollingCumReturn_{TARGET}_{WINDOW}'] - df[f'RollingCumReturn_{BASE}_{WINDOW}']
    if 'BTC' in TARGET:
        diff = np.clip(diff, -2, 2)
    latest_diff = diff.values[-1]
    latest_quantile = np.searchsorted(np.sort(diff.values), latest_diff, side='right') / len(diff) * 100

    # Plot
    ax.plot(df.index, diff, label=f'{TARGET} − {BASE}', color=f'C{idx % 10}')
    ax.axhline(0, color='black', linestyle='--', alpha=0.7)
    ax.set_title(f'{TARGET} − {BASE} ({WINDOW}-Day) | pct {latest_quantile:.1f}%', fontsize=11)
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling CumReturn Diff")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

# ===== Adjust Layout =====
for j in range(len(PAIRS), len(axs)):
    axs[j].axis('off')  # hide extra subplots if any

plt.tight_layout()
plt.show()
