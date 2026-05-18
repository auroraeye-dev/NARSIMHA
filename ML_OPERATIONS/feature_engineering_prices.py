import numpy as np
import pandas as pd
from pprint import pprint

# ==============================
# LOAD DATA
# ==============================
BASE = "/Users/SatvikMishra/Desktop/ROUND02/ROUND_2"

df_day_neg1 = pd.read_csv(f"{BASE}/prices_round_2_day_-1.csv", sep=';')
df_day_0    = pd.read_csv(f"{BASE}/prices_round_2_day_0.csv",  sep=';')
df_day_1    = pd.read_csv(f"{BASE}/prices_round_2_day_1.csv",  sep=';')

# ==============================
# SPLIT BY PRODUCT
# ==============================
def split_products(df):
    return {
        "ASH_COATED_OSMIUM":    df[df['product'] == 'ASH_COATED_OSMIUM'].copy(),
        "INTARIAN_PEPPER_ROOT": df[df['product'] == 'INTARIAN_PEPPER_ROOT'].copy()
    }

day_neg1 = split_products(df_day_neg1)
day_0    = split_products(df_day_0)
day_1    = split_products(df_day_1)

# ==============================
# FEATURE EXTRACTION (UPGRADED)
# ==============================
def extract_price_features(df):
    df = df.copy().sort_values('timestamp').reset_index(drop=True)

    # --- CLEAN: drop rows where there's no valid book ---
    # If bid_price_1 or ask_price_1 is NaN, mid is meaningless
    # Also drop mid_price == 0 (artifact of one-sided book)
    valid = df['bid_price_1'].notna() & df['ask_price_1'].notna() & (df['mid_price'] > 0)
    n_dropped = (~valid).sum()
    df = df[valid].reset_index(drop=True)

    # --- Mid price & spread ---
    df['mid'] = (df['bid_price_1'] + df['ask_price_1']) / 2
    df['spread'] = df['ask_price_1'] - df['bid_price_1']

    # --- Total volumes (L1+L2+L3, treating NaN as 0) ---
    bid_vols = df[['bid_volume_1','bid_volume_2','bid_volume_3']].fillna(0)
    ask_vols = df[['ask_volume_1','ask_volume_2','ask_volume_3']].fillna(0)
    df['bid_vol_total'] = bid_vols.sum(axis=1)
    df['ask_vol_total'] = ask_vols.sum(axis=1)

    # --- Imbalance ---
    denom = df['bid_vol_total'] + df['ask_vol_total']
    df['imbalance'] = np.where(
        denom == 0, 0,
        (df['bid_vol_total'] - df['ask_vol_total']) / denom
    )

    # --- Microprice ---
    denom2 = df['bid_volume_1'] + df['ask_volume_1']
    df['microprice'] = np.where(
        (denom2 == 0) | denom2.isna(),
        df['mid'],
        (df['bid_price_1'] * df['ask_volume_1'] +
         df['ask_price_1'] * df['bid_volume_1']) / denom2
    )

    # --- Price DIFFERENCES (raw ticks, not log returns) ---
    # This catches drift better than log returns for small moves
    df['diff'] = df['mid'].diff()

    # --- LINEAR TREND DETECTOR (critical for trend products) ---
    # Fit mid ~ a + b * timestamp
    ts = df['timestamp'].values.astype(float)
    mid = df['mid'].values
    if len(ts) > 10:
        slope, intercept = np.polyfit(ts, mid, 1)
        # R^2 of the linear fit
        fitted = slope * ts + intercept
        ss_res = np.sum((mid - fitted) ** 2)
        ss_tot = np.sum((mid - mid.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        # Total drift = slope * (max_ts - min_ts)
        total_drift = slope * (ts.max() - ts.min())
    else:
        slope = 0; intercept = mid.mean() if len(mid) else 0
        r_squared = 0; total_drift = 0

    # --- START vs END price (robust edge anchors) ---
    head_mid = df['mid'].iloc[:100].mean()  if len(df) >= 100 else df['mid'].mean()
    tail_mid = df['mid'].iloc[-100:].mean() if len(df) >= 100 else df['mid'].mean()

    # --- Mean reversion: autocorrelation of diff ---
    # lag1 ~ -0.5 means bid-ask bounce (not real reversion)
    # lag1 < 0 AND lag5 < 0 means real mean reversion
    ac_lag1  = df['diff'].autocorr(lag=1)  if len(df) > 2  else np.nan
    ac_lag5  = df['diff'].autocorr(lag=5)  if len(df) > 6  else np.nan
    ac_lag20 = df['diff'].autocorr(lag=20) if len(df) > 21 else np.nan

    # --- Volatility: tick-size standard deviation of diff ---
    # Much more interpretable than log return vol for these price levels
    tick_vol = df['diff'].std()

    # --- DETRENDED volatility (remove linear drift first, then measure noise) ---
    detrended = mid - (slope * ts + intercept)
    detrended_std = detrended.std()

    # --- Microprice signal (leading indicator of next mid) ---
    df['mp_edge'] = df['microprice'] - df['mid']

    return {
        # Cleaning diagnostic
        "rows_dropped_invalid": int(n_dropped),
        "rows_used":            int(len(df)),

        # Spread
        "spread_mean":       round(float(df['spread'].mean()), 3),
        "spread_std":        round(float(df['spread'].std()),  3),
        "spread_mode":       int(df['spread'].mode().iloc[0]) if len(df) else None,

        # Price level anchors
        "mid_mean":          round(float(df['mid'].mean()), 2),
        "mid_std":           round(float(df['mid'].std()),  2),
        "head_mid_first100": round(float(head_mid), 2),
        "tail_mid_last100":  round(float(tail_mid), 2),
        "start_to_end_drift":round(float(tail_mid - head_mid), 2),

        # TREND - THE HIDDEN SIGNAL
        "linear_slope_per_tick":  float(slope),
        "linear_slope_per_1000ts":round(float(slope * 1000), 4),
        "linear_total_drift":     round(float(total_drift), 2),
        "linear_r_squared":       round(float(r_squared), 4),

        # Volatility
        "tick_vol_raw":       round(float(tick_vol), 3),
        "tick_vol_detrended": round(float(detrended_std), 3),

        # Mean reversion diagnostics
        "diff_autocorr_lag1":  round(float(ac_lag1),  4) if not np.isnan(ac_lag1)  else None,
        "diff_autocorr_lag5":  round(float(ac_lag5),  4) if not np.isnan(ac_lag5)  else None,
        "diff_autocorr_lag20": round(float(ac_lag20), 4) if not np.isnan(ac_lag20) else None,

        # Order book
        "imbalance_mean":   round(float(df['imbalance'].mean()), 4),
        "imbalance_std":    round(float(df['imbalance'].std()),  4),
        "microprice_bias":  round(float(df['mp_edge'].mean()), 4),
        "microprice_std":   round(float(df['mp_edge'].std()),  4),
    }

# ==============================
# BUILD STRUCTURED OUTPUT
# ==============================
results_price = {}
products = ["ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"]

for product in products:
    results_price[product] = {
        "day_-1": extract_price_features(day_neg1[product]),
        "day_0":  extract_price_features(day_0[product]),
        "day_1":  extract_price_features(day_1[product])
    }

# ==============================
# OUTPUT
# ==============================
pprint(results_price, sort_dicts=False, width=100)

# ==============================
# CROSS-DAY SUMMARY: does the trend persist day-over-day?
# ==============================
print("\n" + "="*70)
print("CROSS-DAY PATTERN SUMMARY")
print("="*70)
for product in products:
    print(f"\n{product}:")
    print(f"  {'Day':<8}{'StartMid':<12}{'EndMid':<12}{'Drift':<10}{'Slope/1k ts':<14}{'R^2':<8}")
    for day_key in ["day_-1", "day_0", "day_1"]:
        r = results_price[product][day_key]
        print(f"  {day_key:<8}"
              f"{r['head_mid_first100']:<12}"
              f"{r['tail_mid_last100']:<12}"
              f"{r['start_to_end_drift']:<10}"
              f"{r['linear_slope_per_1000ts']:<14}"
              f"{r['linear_r_squared']:<8}")
    # Start-of-day shift between consecutive days
    s_neg1 = results_price[product]['day_-1']['head_mid_first100']
    s_0    = results_price[product]['day_0']['head_mid_first100']
    s_1    = results_price[product]['day_1']['head_mid_first100']
    print(f"  Start-of-day shift: day_-1->day_0 = {s_0 - s_neg1:+.1f}, day_0->day_1 = {s_1 - s_0:+.1f}")
    print(f"  >>> Predicted start-of-day for hidden day_2: {s_1 + (s_1 - s_0):.1f}")