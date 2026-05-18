"""
ETL + strategy backtest for Prosperity 4, Round 2.

Products: ASH_COATED_OSMIUM (stable ~10000), INTARIAN_PEPPER_ROOT (linear drift +1/1000 ts).

Parameters are calibrated to the actual Round 2 data:
  - OSMIUM: mid ~ 9985, spread mode 16 (half 8), pure bid-ask bounce
  - INTARIAN: perfect linear drift (R^2 = 1.0), slope +0.001/ts, spread mode 13-14 (half 7)
              start-of-day shifts +1000 every day => hidden day starts ~14005
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIG
# ==============================================================================
DAY = 0
ROUND_NO = 2
DATA_DIR = "/Users/SatvikMishra/Desktop/ROUND02/ROUND_2"
OUT_DIR  = "/Users/SatvikMishra/Desktop/ROUND_1/DATA_ROUND_1"

# Product-specific constants derived from data analysis (see README)
OSMIUM_FAIR_CONST = 10000        # rounded center; true mean is ~9983 but MM ladder uses 10000
OSMIUM_HALF_SPREAD = 8           # spread mode = 16 => half = 8 (best bid/ask sits at mid +/- 8)
OSMIUM_EMA_ALPHA = 0.002         # half-life ~ 346 ticks; OK for a stable series

INTARIAN_START_DAY_NEG1 = 11005  # empirical start-of-day price
INTARIAN_SLOPE_PER_TICK = 0.001  # R^2 = 1.0 across all 3 days
INTARIAN_HALF_SPREAD = 7         # spread mode ~ 13-14 => half = 7
# For any day d >= -1:  fair(t) = INTARIAN_START_DAY_NEG1 + 1000*(d - (-1)) + 0.001*t
#                                = 11005 + 1000*(d+1) + 0.001*t

LAMBDA_MICRO = 0.5               # microprice-mid adjustment weight (was 1.8 in old code, unsupported)
TICK = 1

POSITION_LIMIT = 80              # Round 2 limit per product


# ==============================================================================
# HELPERS
# ==============================================================================
def intarian_fair_price(day: int, timestamp: pd.Series) -> pd.Series:
    """Drift model: every day starts +1000 above the previous, +0.001 per tick within day."""
    start = INTARIAN_START_DAY_NEG1 + 1000 * (day + 1)
    return start + INTARIAN_SLOPE_PER_TICK * timestamp


def clean_mid(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a reliable mid; drop rows where the book is one-sided or empty."""
    df = df.copy()
    valid = df["bid_price_1"].notna() & df["ask_price_1"].notna()
    df.loc[~valid, ["bid_price_1", "ask_price_1"]] = np.nan
    df["mid_price"] = (df["bid_price_1"] + df["ask_price_1"]) / 2.0
    df["mid_price"] = df["mid_price"].ffill().bfill()
    df["spread"] = df["ask_price_1"] - df["bid_price_1"]
    return df


def compute_microprice(df: pd.DataFrame) -> pd.Series:
    """Volume-weighted mid; falls back to mid when L1 volumes are zero."""
    num = df["ask_price_1"] * df["bid_volume_1"] + df["bid_price_1"] * df["ask_volume_1"]
    den = df["bid_volume_1"] + df["ask_volume_1"]
    mp = np.where((den > 0) & den.notna(), num / den.replace(0, np.nan), df["mid_price"])
    return pd.Series(mp, index=df.index).ffill()


def load_and_split(day: int):
    prc = pd.read_csv(f"{DATA_DIR}/prices_round_{ROUND_NO}_day_{day}.csv", sep=';')
    trd = pd.read_csv(f"{DATA_DIR}/trades_round_{ROUND_NO}_day_{day}.csv", sep=';')

    prc_osmium = prc[prc["product"] == "ASH_COATED_OSMIUM"].copy()
    prc_pepper = prc[prc["product"] == "INTARIAN_PEPPER_ROOT"].copy()
    trd_osmium = trd[trd["symbol"]  == "ASH_COATED_OSMIUM"].copy()
    trd_pepper = trd[trd["symbol"]  == "INTARIAN_PEPPER_ROOT"].copy()

    # Drop counterparty/currency cols and aggregate duplicate-timestamp trades
    def agg_trades(t):
        t = t.drop(columns=["buyer", "seller", "currency"], errors="ignore")
        return t.groupby(["timestamp", "price"], as_index=False).agg({"quantity": "sum"})

    trd_osmium = agg_trades(trd_osmium)
    trd_pepper = agg_trades(trd_pepper)
    return prc_osmium, prc_pepper, trd_osmium, trd_pepper


def merge_and_clean(prc: pd.DataFrame, trd: pd.DataFrame) -> pd.DataFrame:
    # LEFT join: keep every book observation even when no trade happened.
    # If multiple trades in one timestamp, collapse to the volume-weighted avg price & total qty.
    trd_collapsed = (
        trd.groupby("timestamp")
           .apply(lambda g: pd.Series({
               "trade_price": (g["price"] * g["quantity"]).sum() / g["quantity"].sum(),
               "trade_qty":   g["quantity"].sum(),
           }))
           .reset_index()
    )
    merged = prc.merge(trd_collapsed, on="timestamp", how="left")
    merged = clean_mid(merged)
    merged["microprice"] = compute_microprice(merged)
    return merged


# ==============================================================================
# LOAD
# ==============================================================================
prc_osmium, prc_pepper, trd_osmium, trd_pepper = load_and_split(DAY)

merged_osmium = merge_and_clean(prc_osmium, trd_osmium)
merged_pepper = merge_and_clean(prc_pepper, trd_pepper)

merged_osmium.to_csv(f"{OUT_DIR}/merged_ASH_COATED_OSMIUM_day_{DAY}.csv", index=False)
merged_pepper.to_csv(f"{OUT_DIR}/merged_INTARIAN_PEPPER_ROOT_day_{DAY}.csv", index=False)


# ==============================================================================
# FAIR-PRICE MODELS
# ==============================================================================
# --- OSMIUM: EMA of mid + lambda * (microprice - mid) anchored to 10000 ---
merged_osmium["ema"] = merged_osmium["mid_price"].ewm(alpha=OSMIUM_EMA_ALPHA, adjust=False).mean()
merged_osmium["fair_price"] = (
    merged_osmium["ema"]
    + LAMBDA_MICRO * (merged_osmium["microprice"] - merged_osmium["mid_price"])
)

# --- INTARIAN: deterministic linear drift (start + 0.001 * t) ---
merged_pepper["fair_price"] = intarian_fair_price(DAY, merged_pepper["timestamp"])
# Optional: blend a tiny microprice tilt so adverse flow nudges fair
merged_pepper["fair_price"] = (
    merged_pepper["fair_price"]
    + LAMBDA_MICRO * (merged_pepper["microprice"] - merged_pepper["mid_price"])
)


# ==============================================================================
# STRATEGY BACKTEST 1: SCALP (take market orders that cross fair)
# ==============================================================================
def scalp_pnl(df: pd.DataFrame, label: str) -> float:
    """Post limits 1 tick better than the market trade price; earn (fair - fill) per unit."""
    d = df.dropna(subset=["trade_price", "trade_qty", "fair_price"]).copy()
    if d.empty:
        print(f"{label} SCALP: no trades to evaluate")
        return 0.0
    d["scalp_price"] = np.where(d["trade_price"] < d["fair_price"],
                                d["trade_price"] + TICK,
                                d["trade_price"] - TICK)
    d["pnl"] = np.where(d["trade_price"] < d["fair_price"],
                        (d["fair_price"] - d["scalp_price"]) * d["trade_qty"],
                        (d["scalp_price"] - d["fair_price"]) * d["trade_qty"])
    total = d["pnl"].sum()
    print(f"{label} SCALP total hypothetical PnL: {total:.1f} across {len(d)} trades")
    return total


# ==============================================================================
# STRATEGY BACKTEST 2: JOIN BEST QUEUE
# ==============================================================================
def queue_pnl(df: pd.DataFrame, label: str):
    """Assume we join best bid/ask; we get filled only for volume beyond the existing queue."""
    d = df.dropna(subset=["trade_price", "trade_qty", "fair_price",
                          "bid_price_1", "ask_price_1"]).copy()
    if d.empty:
        print(f"{label} QUEUE: no rows to evaluate")
        return 0.0, 0.0

    # Buy side: trade went off at price < fair => there was a seller crossing
    d["qty_beyond_bid_queue"] = np.where(
        d["trade_price"] < d["fair_price"],
        np.maximum(d["trade_qty"] - d["bid_volume_1"].fillna(0), 0),
        0,
    )
    d["profit_at_best_bid"] = d["qty_beyond_bid_queue"] * (d["fair_price"] - d["bid_price_1"])

    d["qty_beyond_ask_queue"] = np.where(
        d["trade_price"] > d["fair_price"],
        np.maximum(d["trade_qty"] - d["ask_volume_1"].fillna(0), 0),
        0,
    )
    d["profit_at_best_ask"] = d["qty_beyond_ask_queue"] * (d["ask_price_1"] - d["fair_price"])

    pb = d["profit_at_best_bid"].sum()
    pa = d["profit_at_best_ask"].sum()
    print(f"{label} QUEUE bid PnL: {pb:.1f}   ask PnL: {pa:.1f}   total: {pb + pa:.1f}")
    return pb, pa


# ==============================================================================
# STRATEGY BACKTEST 3: INTARIAN DIRECTIONAL (ride the drift)
# ==============================================================================
def intarian_drift_pnl(df: pd.DataFrame, position_limit: int = POSITION_LIMIT) -> float:
    """
    Naive upper bound: go max long at tick 0, exit at the last tick.
    Proves the theoretical ceiling of the drift edge.
    """
    d = df.dropna(subset=["ask_price_1", "bid_price_1"]).copy()
    if d.empty:
        return 0.0
    entry_ask = d["ask_price_1"].iloc[0]           # price to buy at
    exit_bid  = d["bid_price_1"].iloc[-1]          # price to sell at
    pnl = position_limit * (exit_bid - entry_ask)
    print(f"INTARIAN drift ceiling: buy {position_limit} @ {entry_ask:.1f}, "
          f"sell @ {exit_bid:.1f}, PnL = {pnl:.0f}")
    return pnl


# Run backtests
print(f"\n=== ASH_COATED_OSMIUM (day {DAY}) ===")
scalp_osmium = scalp_pnl(merged_osmium, "OSMIUM")
queue_b, queue_a = queue_pnl(merged_osmium, "OSMIUM")

print(f"\n=== INTARIAN_PEPPER_ROOT (day {DAY}) ===")
scalp_pepper = scalp_pnl(merged_pepper, "INTARIAN")
queue_pb, queue_pa = queue_pnl(merged_pepper, "INTARIAN")
drift_ceiling  = intarian_drift_pnl(merged_pepper)


# ==============================================================================
# DIAGNOSTIC: how often does market price confirm fair direction?
# ==============================================================================
def direction_confirmation(df: pd.DataFrame, label: str):
    d = df.dropna(subset=["trade_price", "fair_price", "bid_price_1", "ask_price_1"])
    if d.empty:
        return
    below_fair_and_ask_below = ((d["trade_price"] < d["fair_price"]) &
                                (d["ask_price_1"] < d["fair_price"])).sum()
    above_fair_and_bid_above = ((d["trade_price"] > d["fair_price"]) &
                                (d["bid_price_1"] > d["fair_price"])).sum()
    n = len(d)
    print(f"{label} confirm: mkt<fair & ask<fair  {below_fair_and_ask_below} ({below_fair_and_ask_below/n*100:.1f}%) | "
          f"mkt>fair & bid>fair  {above_fair_and_bid_above} ({above_fair_and_bid_above/n*100:.1f}%)")

direction_confirmation(merged_osmium, "OSMIUM")
direction_confirmation(merged_pepper, "INTARIAN")


# ==============================================================================
# PLOTS
# ==============================================================================
def plot_mid_with_fair(df: pd.DataFrame, title: str, offset: int):
    d = df.dropna(subset=["mid_price", "fair_price"])
    plt.figure(figsize=(12, 4))
    plt.plot(d["timestamp"], d["mid_price"], label="Mid", color="steelblue", lw=0.7)
    plt.plot(d["timestamp"], d["fair_price"], label="Fair", color="darkorange", lw=1.2)
    plt.plot(d["timestamp"], d["fair_price"] - offset, ls="--", color="forestgreen",
             label=f"Fair−{offset} (buy band)")
    plt.plot(d["timestamp"], d["fair_price"] + offset, ls="--", color="crimson",
             label=f"Fair+{offset} (sell band)")
    plt.title(f"{title} — day {DAY}")
    plt.xlabel("timestamp"); plt.ylabel("price")
    plt.legend(loc="upper left"); plt.grid(True)
    plt.show()

plot_mid_with_fair(merged_osmium, "ASH_COATED_OSMIUM", OSMIUM_HALF_SPREAD)
plot_mid_with_fair(merged_pepper, "INTARIAN_PEPPER_ROOT", INTARIAN_HALF_SPREAD)


# ==============================================================================
# INVENTORY SIZING CURVE (Beta distribution, properly calibrated)
# ==============================================================================
def beta_pdf(u, alpha, beta):
    if u < 0.0 or u > 1.0:
        return 0.0
    beta_func = math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)
    return (u ** (alpha - 1) * (1.0 - u) ** (beta - 1)) / beta_func


def build_mirrored_beta_distribution(a, b, alphaL, alphaR, steps=100):
    """Descending Beta on [a..b] with a > b (price gets more negative)."""
    if not (a > b):
        raise ValueError("Need a > b for mirrored distribution")
    x_vals = np.linspace(a, b, steps)
    pdf_u = np.array([beta_pdf((a - x) / (a - b), alphaL, alphaR) for x in x_vals])
    pdf_x = pdf_u / abs(b - a)
    cdf = np.zeros_like(pdf_x)
    for i in range(1, steps):
        dx = x_vals[i] - x_vals[i - 1]
        cdf[i] = cdf[i - 1] + 0.5 * (pdf_x[i] + pdf_x[i - 1]) * abs(dx)

    def cdf_at(x):
        if x >= a: return 0.0
        if x <= b: return 1.0
        return float(np.interp(x, x_vals[::-1], cdf[::-1]))

    return x_vals, pdf_x, cdf, cdf_at


def build_ascending_beta_distribution(a, b, alphaL, alphaR, steps=100):
    """Ascending Beta on [a..b] with a < b."""
    if not (a < b):
        raise ValueError("Need a < b for ascending distribution")
    x_vals = np.linspace(a, b, steps)
    pdf_u = np.array([beta_pdf((x - a) / (b - a), alphaL, alphaR) for x in x_vals])
    pdf_x = pdf_u / (b - a)
    cdf = np.zeros_like(pdf_x)
    for i in range(1, steps):
        dx = x_vals[i] - x_vals[i - 1]
        cdf[i] = cdf[i - 1] + 0.5 * (pdf_x[i] + pdf_x[i - 1]) * dx

    def cdf_at(x):
        if x <= a: return 0.0
        if x >= b: return 1.0
        return float(np.interp(x, x_vals, cdf))

    return x_vals, pdf_x, cdf, cdf_at


# Ranges are in (price - fair) space.
# Best bid sits at fair - half_spread. Start scaling in there, max out 5 ticks deeper.
# Half-spreads: OSMIUM=8, INTARIAN=7. Use OSMIUM params here; duplicate for pepper as needed.
BUY_A, BUY_B   = -OSMIUM_HALF_SPREAD, -OSMIUM_HALF_SPREAD - 5    # -8, -13
SELL_A, SELL_B =  OSMIUM_HALF_SPREAD,  OSMIUM_HALF_SPREAD + 5    #  8,  13

ALPHA_L_BUY, ALPHA_R_BUY   = 2.0, 2.0
ALPHA_L_SELL, ALPHA_R_SELL = 2.0, 2.0

x_buy,  pdf_buy,  cdf_buy,  cdf_buy_at  = build_mirrored_beta_distribution(
    BUY_A, BUY_B, ALPHA_L_BUY, ALPHA_R_BUY, steps=100)
x_sell, pdf_sell, cdf_sell, cdf_sell_at = build_ascending_beta_distribution(
    SELL_A, SELL_B, ALPHA_L_SELL, ALPHA_R_SELL, steps=100)


def buy_position(ask_edge, scale=POSITION_LIMIT):
    """ask_edge = ask_price - fair_price. Negative => buy signal. Returns desired long qty."""
    return scale * cdf_buy_at(ask_edge)


def sell_position(bid_edge, scale=POSITION_LIMIT):
    """bid_edge = bid_price - fair_price. Positive => sell signal. Returns desired short qty (negative)."""
    return -scale * cdf_sell_at(bid_edge)


# Quick sanity print
test_ask_edge = -10   # ask is 10 below fair => strong buy
test_bid_edge =  10   # bid is 10 above fair => strong sell
print(f"\nSizing test: ask_edge={test_ask_edge} => buy {buy_position(test_ask_edge):.1f}, "
      f"bid_edge={test_bid_edge} => sell {sell_position(test_bid_edge):.1f}")


# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 70)
print(f"SUMMARY  day={DAY}  (remember: scoring day PnL ~ backtest / 3)")
print("=" * 70)
print(f"OSMIUM    scalp: {scalp_osmium:>10.1f}   queue: {queue_b + queue_a:>10.1f}")
print(f"INTARIAN  scalp: {scalp_pepper:>10.1f}   queue: {queue_pb + queue_pa:>10.1f}   "
      f"drift ceiling: {drift_ceiling:>10.0f}")