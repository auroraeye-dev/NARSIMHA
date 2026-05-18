import pandas as pd
import numpy as np

day = -2
round_no = 1

prc_df = pd.read_csv(f"file path", sep=';')
trades_df = pd.read_csv(f"file path", sep=';')

#################################################################ASH_COATED OSMIUM#####################################################
###########################################################################################################################################

# Filter per product
prc_osmium = prc_df[prc_df['product'] == "ASH_COATED_OSMIUM"]
prc_pepper = prc_df[prc_df['product'] == "INTARIAN_PEPPER_ROOT"]
trades_osmium = trades_df[trades_df['symbol'] == "ASH_COATED_OSMIUM"]
trades_pepper = trades_df[trades_df['symbol'] == "INTARIAN_PEPPER_ROOT"]

# Clean and aggregate trades
for t_df in [trades_osmium, trades_pepper]:
    t_df.drop(columns=["buyer", "seller", "currency"], inplace=True)
    t_df = t_df.groupby(['timestamp', 'price'], as_index=False).agg({'quantity': 'sum'})
    

# Merge price + trade data on timestamp
merged_osmium = prc_osmium.merge(trades_osmium[['timestamp', 'price', 'quantity']], on='timestamp', how='left')
merged_pepper = prc_pepper.merge(trades_pepper[['timestamp', 'price', 'quantity']], on='timestamp', how='left')

# Save merged data
merged_osmium.to_csv(f"file pathmerged_osmium_day_{day}.csv", index=False)
merged_pepper.to_csv(f"file pathmerged_pepper_day_{day}.csv", index=False)

# fair price thing

merged_osmium["mid_price"] = (merged_osmium["bid_price_1"] + merged_osmium["ask_price_1"]) / 2
tick = 1

merged_osmium['scalp_price'] = np.where(
    merged_osmium['price'] < merged_osmium["mid_price"],
    merged_osmium['price'] + tick,
    merged_osmium['price'] - tick
)
# Clean mid price first
merged_osmium["mid_price"] = np.where(
    (merged_osmium["bid_price_1"] > 0) & (merged_osmium["ask_price_1"] > 0),
    (merged_osmium["bid_price_1"] + merged_osmium["ask_price_1"]) / 2,
    np.nan
)
merged_osmium["mid_price"] = merged_osmium["mid_price"].ffill()

# EMA
merged_osmium["ema"] = merged_osmium["mid_price"].ewm(alpha=0.002).mean()

# Microprice
merged_osmium["microprice"] = (
    merged_osmium["ask_price_1"] * merged_osmium["bid_volume_1"] +
    merged_osmium["bid_price_1"] * merged_osmium["ask_volume_1"]
) / (merged_osmium["bid_volume_1"] + merged_osmium["ask_volume_1"])

# FINAL FAIR PRICE (vector)
lambda_ = 1.8
merged_osmium["fair_price"] = (
    merged_osmium["ema"] +
    lambda_ * (merged_osmium["microprice"] - merged_osmium["mid_price"])
)

merged_osmium['expected_profit_scalp'] = np.where(
    merged_osmium['price'] < merged_osmium["fair_price"],
    (merged_osmium["fair_price"] - merged_osmium['scalp_price']) * merged_osmium['quantity'],
    (merged_osmium['scalp_price'] - merged_osmium["fair_price"]) * merged_osmium['quantity']
)

total_hypo_pnl = merged_osmium['expected_profit_scalp'].sum()
print(f"Total expected profit posting aggressive limits: {total_hypo_pnl}")

# Now let's calculate the profit from joining the best‐bid or best‐ask queue

merged_osmium['remaining_qty_at_best_bid'] = np.where(
    merged_osmium['price'] < merged_osmium["fair_price"],
    np.maximum(
        merged_osmium['quantity'] - merged_osmium['bid_volume_1'],
        0
    ),
    0
)

merged_osmium['profit_at_best_bid'] = (
    merged_osmium['remaining_qty_at_best_bid'] *
    (merged_osmium["fair_price"] - merged_osmium['bid_price_1'])
)

merged_osmium['remaining_qty_at_best_ask'] = np.where(
    merged_osmium['price'] > merged_osmium["fair_price"],
    np.maximum(
        merged_osmium['quantity'] - merged_osmium['ask_volume_1'],
        0
    ),
    0
)

merged_osmium['profit_at_best_ask'] = (
    merged_osmium['remaining_qty_at_best_ask'] *
    (merged_osmium['ask_price_1'] - merged_osmium["fair_price"])
)


total_profit_best_bid = merged_osmium['profit_at_best_bid'].sum()
print(f"Total profit joining the best‐bid queue: {total_profit_best_bid}")

total_profit_best_ask = merged_osmium['profit_at_best_ask'].sum()
print(f"Total profit joining the best‐ask queue: {total_profit_best_ask}")

# Compare the two strategies
print("Scalp strategy P&L:         ", total_hypo_pnl)
print("Total P&L joining the QUEUE: ", total_profit_best_bid+total_profit_best_ask)

 # Let's also check how often the market order price is on the same side of the fair price as the best bid/ask
fair_price = 10000


cond_a = (
    (merged_osmium['price'] < fair_price) &
    (merged_osmium['ask_price_1']      < fair_price)
)
freq_a = cond_a.sum()


cond_b = (
    (merged_osmium['price'] > fair_price) &
    (merged_osmium['bid_price_1']       > fair_price)
)
freq_b = cond_b.sum()


total_events = len(merged_osmium)
pct_a = freq_a / total_events * 100
pct_b = freq_b / total_events * 100

print(f"Events where market < fair & best-ask < fair: {freq_a} ({pct_a:.1f}%)")
print(f"Events where market > fair & best-bid > fair: {freq_b} ({pct_b:.1f}%)")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(f"file path/merged_ASH_COATED_OSMIUM_day_-2.csv")
df["spread"] = df["ask_price_1"] - df["bid_price_1"]

plt.plot(df["timestamp"], df["mid_price"])
plt.title("Mid Price over Time")
plt.xlabel("Timestamp"); plt.ylabel("Mid Price")
plt.show()
alpha_ema = 0.002
df["mid_price"] = np.where(
    (df["bid_price_1"] > 0) & (df["ask_price_1"] > 0),
    (df["bid_price_1"] + df["ask_price_1"]) / 2,
    np.nan
)
df["mid_price"] = df["mid_price"].ffill()
df["ema"]       = df["mid_price"].ewm(alpha=alpha_ema, adjust=False).mean()
offset = 12
df["ema_minus"] = df["ema"] - offset
df["ema_plus"]  = df["ema"] + offset

ema_current = df["ema"].iloc[-1]


plt.figure(figsize=(12,4))
plt.plot(df["timestamp"], df["mid_price"],   label="Mid Price", color="steelblue")
plt.plot(df["timestamp"], df["ema"],         label="EMA",       color="darkorange", lw=2)
plt.plot(df["timestamp"], df["ema_minus"],   label=f"EMA−{offset}", color="forestgreen", ls="--")
plt.plot(df["timestamp"], df["ema_plus"],    label=f"EMA+{offset}", color="crimson",    ls="--")
plt.title("ash coated osmium Mid Price with EMA ± Offset Bands")
plt.xlabel("Timestamp"); plt.ylabel("Price")
plt.legend(loc="upper left"); plt.grid(True)
plt.show()

df['deviation'] = df['mid_price'] - df['ema']

plt.figure(figsize=(12,4))
plt.plot(df['timestamp'], df['deviation'], color='purple', lw=1)
plt.axhline(0, color='gray', ls='--')
plt.title('ash coated osmium: (Mid Price − EMA) Deviation Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Deviation')
plt.grid(True)
plt.show()
import math
import numpy as np
import matplotlib.pyplot as plt

def beta_pdf(u, alpha, beta):
    """
    Returns the value of the Beta(alpha,beta) PDF at u, for 0<=u<=1.
    Outside [0,1], returns 0.
    """
    if u < 0.0 or u > 1.0:
        return 0.0

    # Beta function in the denominator
    beta_func = math.gamma(alpha)*math.gamma(beta)/math.gamma(alpha + beta)

    # Numerator: u^(alpha-1) * (1-u)^(beta-1)
    return (u**(alpha - 1) * (1.0 - u)**(beta - 1)) / beta_func

# Parameters
alphaL = 2.0   # shape for left tail
alphaR = 4.0   # shape for right tail
ema    = df['ema'].iloc[-1]
a      = 0 - 12.0      # left endpoint
b      = 0 - 12.0 - 5.0  # right endpoint of mirrored interval

# Make a grid of x-values
x_vals = np.linspace(a, b, 200)

# Convert each x to u in [0,1]
u_vals = (a - x_vals) / (a - b)

# Compute the Beta PDF on [a,b] manually
pdf_u = np.array([beta_pdf(u, alphaL, alphaR) for u in u_vals])

# Scale for [a,b]: pdf_x = pdf_u / |b-a|
pdf_x = pdf_u / abs(b - a)

# Plot
plt.plot(x_vals, pdf_x)
plt.title(f"Shifted Beta PDF on [{b:.1f}, {a:.1f}] (α={alphaL}, β={alphaR})")
plt.xlabel("Price")
plt.ylabel("PDF(x)")
plt.show()

import math
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
#                  1) MIRRORED BETA ON [a..b] WHEN a > b (BUY SIDE)
###############################################################################
def beta_pdf(u, alpha, beta):
    """
    Returns Beta(alpha,beta) PDF at u in [0,1].
    Outside [0,1], returns 0.
    """
    if u < 0.0 or u > 1.0:
        return 0.0
    beta_func = math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)
    return (u**(alpha - 1) * (1.0 - u)**(beta - 1)) / beta_func

def build_mirrored_beta_distribution(a, b, alphaL, alphaR, steps=100):
    """
    Builds a Beta-based distribution on [a..b] when a > b (descending).
    Returns x_vals_desc, pdf_desc, cdf_desc, cdf_at(x).
    """
    if not (a > b):
        raise ValueError("For mirrored distribution, expect a > b.")
    x_vals_desc = np.linspace(a, b, steps)
    def x_to_u(x):
        return (a - x) / (a - b)
    pdf_u_desc = np.array([beta_pdf(x_to_u(x), alphaL, alphaR) for x in x_vals_desc])
    pdf_x_desc = pdf_u_desc / abs(b - a)
    cdf_desc = np.zeros_like(pdf_x_desc)
    for i in range(1, steps):
        dx = x_vals_desc[i] - x_vals_desc[i - 1]
        area = 0.5 * (pdf_x_desc[i] + pdf_x_desc[i - 1]) * abs(dx)
        cdf_desc[i] = cdf_desc[i - 1] + area
    def cdf_at(x):
        if x >= a:
            return 0.0
        if x <= b:
            return 1.0
        x_asc = x_vals_desc[::-1]
        cdf_asc = cdf_desc[::-1]
        return np.interp(x, x_asc, cdf_asc)
    return x_vals_desc, pdf_x_desc, cdf_desc, cdf_at

###############################################################################
#      2) ASCENDING BETA ON [a..b] WHEN a < b (SELL SIDE)
###############################################################################
def build_ascending_beta_distribution(a, b, alphaL, alphaR, steps=100):
    """
    Standard Beta distribution on [a..b], a<b, ascending array.
    Returns x_vals, pdf_x, cdf_x, cdf_at(x).
    """
    if not (a < b):
        raise ValueError("For ascending distribution, expect a < b.")
    x_vals = np.linspace(a, b, steps)
    def x_to_u(x):
        return (x - a) / (b - a)
    pdf_u = np.array([beta_pdf(x_to_u(xx), alphaL, alphaR) for xx in x_vals])
    pdf_x = pdf_u / (b - a)
    cdf_x = np.zeros_like(pdf_x)
    for i in range(1, steps):
        dx = x_vals[i] - x_vals[i - 1]
        area = 0.5 * (pdf_x[i] + pdf_x[i - 1]) * dx
        cdf_x[i] = cdf_x[i - 1] + area
    def cdf_at(x):
        if x <= a:
            return 0.0
        if x >= b:
            return 1.0
        return np.interp(x, x_vals, cdf_x)
    return x_vals, pdf_x, cdf_x, cdf_at

###############################################################################
#            3) BUILD THE "BUY" SIDE (MIRRORED) on [-12, -17]
###############################################################################
alphaL_buy, alphaR_buy = 2.0, 2.0
buy_a, buy_b = -12, -17  # a > b
x_vals_buy, pdf_buy, cdf_buy, cdf_buy_at = build_mirrored_beta_distribution(
    buy_a, buy_b, alphaL_buy, alphaR_buy, steps=100
)

def buy_position(ask_price, scale=50):
    """
    Positive inventory for buying:
      ask=200 => cdf_at(200)=1 => buy=+60
      ask=330 => cdf_at(330)=0 => buy=0
    => buy_position = scale * cdf_at(ask_price)
    """
    return scale * cdf_buy_at(ask_price)

###############################################################################
#           4) BUILD THE "SELL" SIDE (ASCENDING) on [12, 17]
###############################################################################
alphaL_sell, alphaR_sell = 2.0, 2.0
sell_a, sell_b = 12, 17  # a < b
x_vals_sell, pdf_sell, cdf_sell, cdf_sell_at = build_ascending_beta_distribution(
    sell_a, sell_b, alphaL_sell, alphaR_sell, steps=100
)

def sell_position(bid_price, scale=50):
    """
    Negative inventory for selling:
      bid=390 => cdf_at(390)=0 => sell=0
      bid=500 => cdf_at(500)=1 => sell=-60
    => sell_position = -scale * cdf_at(bid_price)
    """
    return -scale * cdf_sell_at(bid_price)

###############################################################################
#                 5) QUICK TEST
###############################################################################
test_ask = 15
test_bid = 13
inv_buy_side = buy_position(test_ask, scale=50)
inv_sell_side = sell_position(test_bid, scale=50)
inv_total = inv_buy_side + inv_sell_side
print(f"ask={test_ask}, bid={test_bid}")
print(f"buy side = {inv_buy_side:.2f}, sell side = {inv_sell_side:.2f}, total={inv_total:.2f}")

###############################################################################
#                6) PLOTTING
###############################################################################
# Plot buy distribution
plt.figure()
plt.plot(x_vals_buy, pdf_buy, label="Buy PDF [ema-12, ema-12-5] (mirrored)")
plt.plot(x_vals_buy, cdf_buy, label="Buy CDF [ema-12, ema-12-5]")
plt.title("MIRRORED 'Buy' Dist (Ask Price in [ema-12, ema-12-5])")
plt.xlabel("Ask Price")
plt.legend()
plt.grid(True)
plt.show()

# Plot sell distribution
plt.figure()
plt.plot(x_vals_sell, pdf_sell, label="Sell PDF [ema+12, ema+12+5]")
plt.plot(x_vals_sell, cdf_sell, label="Sell CDF [ema+12, ema+12+5]")
plt.title("ASCENDING 'Sell' Dist (Bid Price in [ema+12, ema+12+5])")
plt.xlabel("Bid Price")
plt.legend()
plt.grid(True)
plt.show()
#################################################################INTARIAN PEPPER ROOT#####################################################
###########################################################################################################################################
prc_df_INTARIAN_PEPPER_ROOT = prc_df[prc_df['product'] == "INTARIAN_PEPPER_ROOT"]
trades_df_INTARIAN_PEPPER_ROOT = trades_df[trades_df["symbol"] == "INTARIAN_PEPPER_ROOT"]
trades_df_INTARIAN_PEPPER_ROOT = trades_df_INTARIAN_PEPPER_ROOT.drop(columns=["buyer","seller","currency"])

trades_df_INTA = (
    trades_df_INTARIAN_PEPPER_ROOT
    .groupby(['timestamp', 'price'], as_index=False)
    .agg({'quantity': 'sum'})
)
merged_INTARIAN_PEPPER_ROOT_df = prc_df_INTARIAN_PEPPER_ROOT.merge(
    trades_df_INTARIAN_PEPPER_ROOT[['timestamp', 'price', 'quantity']],
    on='timestamp',
    how='inner')

#SAVE MERGED DATA
merged_INTARIAN_PEPPER_ROOT_df.to_csv(f"file path/merged_pepper_day_{day}.csv", index=False)

# Filter per product
prc_INTARIAN_PEPPER_ROOT = prc_df[prc_df['product'] == "INTARIAN_PEPPER_ROOT"]
prc_ASH_COATED_OSMIUM = prc_df[prc_df['product'] == "ASH_COATED_OSMIUM"]
trades_INTARIAN_PEPPER_ROOT = trades_df[trades_df['symbol'] == "INTARIAN_PEPPER_ROOT"]
trades_ASH_COATED_OSMIUM = trades_df[trades_df['symbol'] == "ASH_COATED_OSMIUM"]

# Clean and aggregate trades
for t_df in [trades_INTARIAN_PEPPER_ROOT, trades_ASH_COATED_OSMIUM]:
    t_df.drop(columns=["buyer", "seller", "currency"], inplace=True)
    t_df = t_df.groupby(['timestamp', 'price'], as_index=False).agg({'quantity': 'sum'})
    

# Merge price + trade data on timestamp
merged_INTARIAN_PEPPER_ROOT = prc_INTARIAN_PEPPER_ROOT.merge(trades_INTARIAN_PEPPER_ROOT[['timestamp', 'price', 'quantity']], on='timestamp', how='inner')
merged_ASH_COATED_OSMIUM = prc_ASH_COATED_OSMIUM.merge(trades_ASH_COATED_OSMIUM[['timestamp', 'price', 'quantity']], on='timestamp', how='inner')

# Save merged data
merged_INTARIAN_PEPPER_ROOT.to_csv(f"/file path/merged_INTARIAN_PEPPER_ROOT_day_{day}.csv", index=False)
merged_ASH_COATED_OSMIUM.to_csv(f"file path/merged_ASH_COATED_OSMIUM_day_{day}.csv", index=False)

# fair price thing

fair_price = 10500
tick = 1

merged_INTARIAN_PEPPER_ROOT['scalp_price'] = np.where(
    merged_INTARIAN_PEPPER_ROOT['price'] < fair_price,
    merged_INTARIAN_PEPPER_ROOT['price'] + tick,
    merged_INTARIAN_PEPPER_ROOT['price'] - tick
)

merged_INTARIAN_PEPPER_ROOT['expected_profit_scalp'] = np.where(
    merged_INTARIAN_PEPPER_ROOT['price'] < fair_price,
    (fair_price - merged_INTARIAN_PEPPER_ROOT['scalp_price']) * merged_INTARIAN_PEPPER_ROOT['quantity'],
    (merged_INTARIAN_PEPPER_ROOT['scalp_price'] - fair_price) * merged_INTARIAN_PEPPER_ROOT['quantity']
)

total_hypo_pnl = merged_INTARIAN_PEPPER_ROOT['expected_profit_scalp'].sum()
print(f"Total expected profit posting aggressive limits: {total_hypo_pnl}")

# Now let's calculate the profit from joining the best‐bid or best‐ask queue

merged_INTARIAN_PEPPER_ROOT['remaining_qty_at_best_bid'] = np.where(
    merged_INTARIAN_PEPPER_ROOT['price'] < fair_price,
    np.maximum(
        merged_INTARIAN_PEPPER_ROOT['quantity'] - merged_INTARIAN_PEPPER_ROOT['bid_volume_1'],
        0
    ),
    0
)

merged_INTARIAN_PEPPER_ROOT['profit_at_best_bid'] = (
    merged_INTARIAN_PEPPER_ROOT['remaining_qty_at_best_bid'] *
    (fair_price - merged_INTARIAN_PEPPER_ROOT['bid_price_1'])
)

merged_INTARIAN_PEPPER_ROOT['remaining_qty_at_best_ask'] = np.where(
    merged_INTARIAN_PEPPER_ROOT['price'] > fair_price,
    np.maximum(
        merged_INTARIAN_PEPPER_ROOT['quantity'] - merged_INTARIAN_PEPPER_ROOT ['ask_volume_1'],
        0
    ),
    0
)

merged_INTARIAN_PEPPER_ROOT['profit_at_best_ask'] = (
    merged_INTARIAN_PEPPER_ROOT['remaining_qty_at_best_ask'] *
    (merged_INTARIAN_PEPPER_ROOT['ask_price_1'] - fair_price)
)


total_profit_best_bid = merged_INTARIAN_PEPPER_ROOT['profit_at_best_bid'].sum()
print(f"Total profit joining the best‐bid queue: {total_profit_best_bid}")

total_profit_best_ask = merged_INTARIAN_PEPPER_ROOT['profit_at_best_ask'].sum()
print(f"Total profit joining the best‐ask queue: {total_profit_best_ask}")

# Compare the two strategies
print("Scalp strategy P&L:         ", total_hypo_pnl)
print("Total P&L joining the QUEUE: ", total_profit_best_bid+total_profit_best_ask)

 # Let's also check how often the market order price is on the same side of the fair price as the best bid/ask
fair_price = 10500


cond_a = (
    (merged_INTARIAN_PEPPER_ROOT['price'] < fair_price) &
    (merged_INTARIAN_PEPPER_ROOT['ask_price_1']      < fair_price)
)
freq_a = cond_a.sum()


cond_b = (
    (merged_INTARIAN_PEPPER_ROOT['price'] > fair_price) &
    (merged_INTARIAN_PEPPER_ROOT['bid_price_1']       > fair_price)
)
freq_b = cond_b.sum()


total_events = len(merged_INTARIAN_PEPPER_ROOT)
pct_a = freq_a / total_events * 100
pct_b = freq_b / total_events * 100

print(f"Events where market < fair & best-ask < fair: {freq_a} ({pct_a:.1f}%)")
print(f"Events where market > fair & best-bid > fair: {freq_b} ({pct_b:.1f}%)")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df= pd.read_csv(f"file path/merged_INTARIAN_PEPPER_ROOT_day_-2.csv")
df["spread"] = df["ask_price_1"] - df["bid_price_1"]

plt.figure(figsize=(12,4))
plt.plot(df["timestamp"], df["mid_price"])
plt.title("Mid Price over Time")
plt.xlabel("Timestamp"); plt.ylabel("Mid Price")
plt.show()
