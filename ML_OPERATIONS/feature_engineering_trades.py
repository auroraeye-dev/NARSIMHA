import pandas as pd
import numpy as np
from pprint import pprint

# ==============================
# FEATURE EXTRACTION (UPGRADED)
# ==============================
def extract_trade_features(df, prices_df=None):
    df = df.copy().sort_values('timestamp').reset_index(drop=True)

    # --- Drop zero/NaN price rows (data artifacts) ---
    df = df[df['price'] > 0].reset_index(drop=True)
    if len(df) == 0:
        return {"error": "no valid trades"}

    # --- Basic volume stats ---
    avg_trade_size = df['quantity'].mean()
    total_volume   = df['quantity'].sum()
    vol_per_ts     = df.groupby('timestamp')['quantity'].sum()

    # --- Large trade threshold ---
    threshold = df['quantity'].quantile(0.9)
    df['is_large'] = df['quantity'] > threshold

    # --- Price differences (raw ticks, not log returns) ---
    df['price_diff'] = df['price'].diff()
    trade_vol_ticks  = df['price_diff'].std()

    # --- Detrend trade prices by timestamp (catches drift products) ---
    ts = df['timestamp'].values.astype(float)
    prices = df['price'].values
    if len(ts) > 10 and ts.std() > 0:
        slope, intercept = np.polyfit(ts, prices, 1)
        fitted = slope * ts + intercept
        ss_res = np.sum((prices - fitted) ** 2)
        ss_tot = np.sum((prices - prices.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        detrended_std = (prices - fitted).std()
    else:
        slope, intercept, r2, detrended_std = 0, prices.mean(), 0, prices.std()

    # --- DIRECTION using QUOTE MIDPOINT (more accurate than tick rule) ---
    # If prices_df provided, compare trade price to contemporaneous mid
    # otherwise fall back to tick rule
    if prices_df is not None and len(prices_df) > 0:
        # For each trade, find the mid price at (or just before) that timestamp
        pdf = prices_df.copy()
        pdf = pdf[(pdf['bid_price_1'].notna()) & (pdf['ask_price_1'].notna())].copy()
        pdf['mid'] = (pdf['bid_price_1'] + pdf['ask_price_1']) / 2
        pdf = pdf.sort_values('timestamp')[['timestamp','mid']]
        # merge_asof: for each trade ts, get last known mid at or before
        df_m = pd.merge_asof(df.sort_values('timestamp'), pdf,
                             on='timestamp', direction='backward')
        df_m['direction'] = np.sign(df_m['price'] - df_m['mid']).fillna(0)
        df = df_m
    else:
        df['direction'] = np.sign(df['price_diff']).replace(0, np.nan).fillna(0)

    df['signed_volume'] = df['quantity'] * df['direction']

    # --- Order flow (rolling) ---
    df['order_flow_20'] = df['signed_volume'].rolling(20, min_periods=1).sum()
    df['order_flow_100'] = df['signed_volume'].rolling(100, min_periods=1).sum()

    # --- Buyer/seller identity - IMPORTANT for Prosperity (some bots signal) ---
    buyers_top = df['buyer'].value_counts().head(5).to_dict() if 'buyer' in df.columns else {}
    sellers_top = df['seller'].value_counts().head(5).to_dict() if 'seller' in df.columns else {}

    # --- Net flow per buyer (who is accumulating vs distributing?) ---
    if 'buyer' in df.columns and df['buyer'].notna().any():
        net_by_buyer  = df.groupby('buyer')['quantity'].sum().sort_values(ascending=False).head(5).to_dict()
        net_by_seller = df.groupby('seller')['quantity'].sum().sort_values(ascending=False).head(5).to_dict()
    else:
        net_by_buyer, net_by_seller = {}, {}

    # --- Future return (next trade) ---
    df['future_diff'] = df['price'].shift(-1) - df['price']
    df['future_return'] = df['future_diff'] / df['price']

    # --- Adverse selection: did price move AGAINST the taker after their trade? ---
    # Taker buys (direction=+1) and price goes UP after => they got ahead, GOOD for them / BAD for MM
    df['adverse_for_mm'] = (
        df['is_large'] & (
            ((df['direction'] ==  1) & (df['future_diff'] > 0)) |
            ((df['direction'] == -1) & (df['future_diff'] < 0))
        )
    )
    adverse_rate = df['adverse_for_mm'].mean()

    # --- Price impact (winsorized) ---
    df['impact'] = df['future_return'] / (df['quantity'] + 1e-9)
    df['impact'] = df['impact'].clip(
        lower=df['impact'].quantile(0.01),
        upper=df['impact'].quantile(0.99)
    )

    return {
        # Volume
        "n_trades":         int(len(df)),
        "avg_trade_size":   round(float(avg_trade_size), 3),
        "total_volume":     int(total_volume),
        "large_threshold":  float(threshold),
        "trades_per_ts":    round(float(vol_per_ts.mean()), 3),

        # Price behavior
        "trade_price_mean":     round(float(df['price'].mean()), 2),
        "trade_vol_ticks":      round(float(trade_vol_ticks), 3),
        "trade_slope_per_1k":   round(float(slope * 1000), 4),
        "trade_linear_r2":      round(float(r2), 4),
        "trade_detrended_std":  round(float(detrended_std), 3),

        # Order flow
        "order_flow_20_mean":  round(float(df['order_flow_20'].mean()), 2),
        "order_flow_20_std":   round(float(df['order_flow_20'].std()), 2),
        "order_flow_100_mean": round(float(df['order_flow_100'].mean()), 2),

        # Impact & adverse selection
        "adverse_rate": round(float(adverse_rate), 4),
        "impact_mean":  float(df['impact'].mean()),
        "impact_std":   float(df['impact'].std()),

        # Buyer/seller signatures (THIS IS WHERE HIDDEN BOT BEHAVIOR LIVES)
        "top_buyers_by_trade_count":  buyers_top,
        "top_sellers_by_trade_count": sellers_top,
        "top_buyers_by_volume":       net_by_buyer,
        "top_sellers_by_volume":      net_by_seller,
    }


# ==============================
# LOAD DATA
# ==============================
BASE = "/Users/SatvikMishra/Desktop/ROUND02/ROUND_2"

# Trades
df_day_neg1_t = pd.read_csv(f"{BASE}/trades_round_2_day_-1.csv", sep=';')
df_day_0_t    = pd.read_csv(f"{BASE}/trades_round_2_day_0.csv",  sep=';')
df_day_1_t    = pd.read_csv(f"{BASE}/trades_round_2_day_1.csv",  sep=';')

# Prices (for merge_asof direction inference)
df_day_neg1_p = pd.read_csv(f"{BASE}/prices_round_2_day_-1.csv", sep=';')
df_day_0_p    = pd.read_csv(f"{BASE}/prices_round_2_day_0.csv",  sep=';')
df_day_1_p    = pd.read_csv(f"{BASE}/prices_round_2_day_1.csv",  sep=';')

# ==============================
# SPLIT PRODUCTS
# ==============================
def split_trades(df):
    return {
        "INTARIAN_PEPPER_ROOT": df[df['symbol']  == 'INTARIAN_PEPPER_ROOT'].copy(),
        "ASH_COATED_OSMIUM":    df[df['symbol']  == 'ASH_COATED_OSMIUM'].copy()
    }

def split_prices(df):
    return {
        "INTARIAN_PEPPER_ROOT": df[df['product'] == 'INTARIAN_PEPPER_ROOT'].copy(),
        "ASH_COATED_OSMIUM":    df[df['product'] == 'ASH_COATED_OSMIUM'].copy()
    }

day_neg1_t = split_trades(df_day_neg1_t); day_neg1_p = split_prices(df_day_neg1_p)
day_0_t    = split_trades(df_day_0_t);    day_0_p    = split_prices(df_day_0_p)
day_1_t    = split_trades(df_day_1_t);    day_1_p    = split_prices(df_day_1_p)

# ==============================
# BUILD RESULTS
# ==============================
results = {}
products = ["INTARIAN_PEPPER_ROOT", "ASH_COATED_OSMIUM"]

for product in products:
    results[product] = {
        "day_-1": extract_trade_features(day_neg1_t[product], day_neg1_p[product]),
        "day_0":  extract_trade_features(day_0_t[product],    day_0_p[product]),
        "day_1":  extract_trade_features(day_1_t[product],    day_1_p[product])
    }

# ==============================
# OUTPUT
# ==============================
pprint(results, sort_dicts=False, width=110)

print("\n" + "="*70)
print("BUYER/SELLER CROSS-DAY SUMMARY (hunting for consistent bots)")
print("="*70)
for product in products:
    print(f"\n{product}:")
    # Which names appear as top buyer/seller across all 3 days?
    all_buyers = set()
    all_sellers = set()
    for day_key in ["day_-1", "day_0", "day_1"]:
        r = results[product][day_key]
        all_buyers.update(r.get("top_buyers_by_volume", {}).keys())
        all_sellers.update(r.get("top_sellers_by_volume", {}).keys())
    print(f"  All top buyers across days:  {sorted(all_buyers)}")
    print(f"  All top sellers across days: {sorted(all_sellers)}")
    for day_key in ["day_-1", "day_0", "day_1"]:
        r = results[product][day_key]
        print(f"  {day_key}: buyers={r.get('top_buyers_by_volume', {})}")
        print(f"  {day_key}: sellers={r.get('top_sellers_by_volume', {})}")