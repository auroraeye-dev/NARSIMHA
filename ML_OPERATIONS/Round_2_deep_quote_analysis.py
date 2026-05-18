"""

If most trades happen at mid ± 8-13
"""
import pandas as pd
import numpy as np

BASE = "/Users/SatvikMishra/Desktop/ROUND02/ROUND_2"

all_devs = []  # OSMIUM trade - mid, across all 3 days
for day in [-1, 0, 1]:
    prc = pd.read_csv(f"{BASE}/prices_round_2_day_{day}.csv", sep=';')
    trd = pd.read_csv(f"{BASE}/trades_round_2_day_{day}.csv", sep=';')

    osm = prc[prc['product'] == 'ASH_COATED_OSMIUM'].copy()
    osm = osm[osm['ask_price_1'].notna() & osm['bid_price_1'].notna()].copy()
    osm['mid'] = (osm['bid_price_1'] + osm['ask_price_1']) / 2
    osm['best_bid'] = osm['bid_price_1']
    osm['best_ask'] = osm['ask_price_1']

    osm_trd = trd[trd['symbol'] == 'ASH_COATED_OSMIUM'].copy()
    merged = osm_trd.merge(
        osm[['timestamp', 'mid', 'best_bid', 'best_ask']],
        on='timestamp', how='inner'
    )
    merged['dev_from_mid'] = merged['price'] - merged['mid']
    # For buy-side trades (price > mid), dev is positive
    # For sell-side trades (price < mid), dev is negative
    merged['abs_dev'] = merged['dev_from_mid'].abs()
    # How far THROUGH the spread did the trade go?
    # If buy: price vs best_ask. If sell: price vs best_bid.
    # A trade AT best_ask = 0 penetration. A trade 5 above best_ask = 5 penetration.
    merged['side'] = np.where(merged['price'] >= merged['mid'], 'buy', 'sell')
    merged['spread_penetration'] = np.where(
        merged['side'] == 'buy',
        merged['price'] - merged['best_ask'],      # how much above best ask
        merged['best_bid'] - merged['price']       # how much below best bid
    )
    all_devs.extend(merged['dev_from_mid'].tolist())

    print(f"\n--- DAY {day} OSMIUM trades ---")
    print(f"  Total trades: {len(merged)}")
    print(f"  Buy side (price > mid): {(merged['side'] == 'buy').sum()}")
    print(f"  Sell side (price < mid): {(merged['side'] == 'sell').sum()}")
    print(f"  |trade - mid| distribution:")
    print(f"    5%={merged['abs_dev'].quantile(0.05):.1f}  "
          f"25%={merged['abs_dev'].quantile(0.25):.1f}  "
          f"50%={merged['abs_dev'].quantile(0.50):.1f}  "
          f"75%={merged['abs_dev'].quantile(0.75):.1f}  "
          f"95%={merged['abs_dev'].quantile(0.95):.1f}")
    print(f"  Trades AT best bid/ask (dev = 7-8):   "
          f"{((merged['abs_dev'] >= 7) & (merged['abs_dev'] <= 8)).sum()}")
    print(f"  Trades at mid ± 9 (1 past best):       "
          f"{((merged['abs_dev'] == 9)).sum()}")
    print(f"  Trades at mid ± 10-12 (deep):          "
          f"{((merged['abs_dev'] >= 10) & (merged['abs_dev'] <= 12)).sum()}")
    print(f"  Trades at mid ± 13+ (extreme):         "
          f"{(merged['abs_dev'] >= 13).sum()}")

# AGGREGATE across all 3 days
print("\n" + "="*70)
print("AGGREGATE ACROSS 3 DAYS — trade price distance from mid")
print("="*70)
all_abs = pd.Series([abs(d) for d in all_devs])
print(f"Total trades: {len(all_abs)}")

# Histogram of distance from mid
bins = list(range(0, 20))
for lo in bins:
    hi = lo + 1
    count = ((all_abs >= lo) & (all_abs < hi)).sum()
    pct = count / len(all_abs) * 100
    bar = '█' * int(pct * 2)
    print(f"  |trade - mid| = {lo:2d}: {count:4d} trades ({pct:5.1f}%) {bar}")

# Key takeaway
print("\n" + "="*70)
print("KEY TAKEAWAY")
print("="*70)
at_7_8 = ((all_abs >= 7) & (all_abs <= 8)).sum()
at_9_12 = ((all_abs >= 9) & (all_abs <= 12)).sum()
at_13p = (all_abs >= 13).sum()
total = len(all_abs)
print(f"Trades at mid ± 7-8  (best bid/ask):  {at_7_8:5d}  ({at_7_8/total*100:.1f}%)")
print(f"Trades at mid ± 9-12 (1-4 past best): {at_9_12:5d}  ({at_9_12/total*100:.1f}%)")
print(f"Trades at mid ± 13+  (very deep):     {at_13p:5d}  ({at_13p/total*100:.1f}%)")
print()
print(f"v6 strategy (mid±7): captures the {at_7_8/total*100:.0f}% shallow flow.")
print(f"If you ALSO post at mid±9, mid±11: capture extra {at_9_12/total*100:.0f}% of flow.")
print(f"Per-unit edge on deep fills: 9-12 ticks vs 7 ticks. MUCH better.")

# POTENTIAL EDGE CALCULATION
# If you post at mid±9 and get filled on half the mid±9 trades,
# edge per fill = 9 ticks, flow = at_9_12 / 2
extra_fills_per_day = at_9_12 / 3 / 2  # half of deep trades become ours
avg_edge_ticks = 10  # rough average of 9, 10, 11, 12
daily_gain = extra_fills_per_day * 5 * avg_edge_ticks  # avg trade size 5
print(f"\nROUGH EDGE PROJECTION (conservative):")
print(f"  Extra deep fills captured per day: ~{extra_fills_per_day:.0f}")
print(f"  Assuming avg fill size 5, edge 10 ticks: +{daily_gain:,.0f}/day")
print(f"  Additional to v6 baseline of ~17k: total ~{17000 + daily_gain:,.0f}/day")
