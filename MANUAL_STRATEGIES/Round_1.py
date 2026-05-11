"""
IMC Prosperity - Manual Auction Solver
=======================================
Batch auction with price-time priority.

Clearing rule:
  1. Clearing price P* = price that maximizes matched volume.
  2. All bids with price >= P* and all asks with price <= P* are eligible.
  3. Matched volume = min(total_demand_at_or_above_clear, total_supply_at_or_below_clear).
  4. Rationing on the long side follows PRICE-TIME priority:
     - Higher-priced bids fill first (for BUY side rationing).
     - At the same price, EXISTING orders fill before ours (we arrived later).
"""

def simulate(bids, asks, my_side, my_price, my_vol):
    """Simulate adding our order and return (my_fill, clear_price)."""
    if my_side == "BUY":
        new_bids = bids + [(my_price, my_vol)]
        new_asks = list(asks)
    else:
        new_bids = list(bids)
        new_asks = asks + [(my_price, my_vol)]

    # Find clearing price that maximizes matched volume
    prices = sorted(set([p for p, _ in new_bids] + [p for p, _ in new_asks]))
    best_matched = -1
    clear = None
    for p in prices:
        d = sum(v for bp, v in new_bids if bp >= p)
        s = sum(v for ap, v in new_asks if ap <= p)
        m = min(d, s)
        if m > best_matched:
            best_matched = m
            clear = p

    total_demand = sum(v for bp, v in new_bids if bp >= clear)
    total_supply = sum(v for ap, v in new_asks if ap <= clear)
    matched = min(total_demand, total_supply)

    if my_side == "BUY":
        if my_price < clear:
            return 0, clear
        # Price-time priority: high price first; existing orders before mine at same price
        bid_list = [(bp, v, False) for bp, v in bids] + [(my_price, my_vol, True)]
        bid_list.sort(key=lambda x: (-x[0], x[2]))  # high price first; False < True -> existing first
        remaining = matched
        my_fill = 0
        for bp, v, is_mine in bid_list:
            if bp < clear or remaining <= 0:
                break
            take = min(v, remaining)
            if is_mine:
                my_fill = take
            remaining -= take
        return my_fill, clear
    else:  # SELL
        if my_price > clear:
            return 0, clear
        ask_list = [(ap, v, False) for ap, v in asks] + [(my_price, my_vol, True)]
        ask_list.sort(key=lambda x: (x[0], x[2]))  # low price first; existing first
        remaining = matched
        my_fill = 0
        for ap, v, is_mine in ask_list:
            if ap > clear or remaining <= 0:
                break
            take = min(v, remaining)
            if is_mine:
                my_fill = take
            remaining -= take
        return my_fill, clear


def find_best(bids, asks, V, buy_fee, sell_fee, max_vol, name):
    """Exhaustive grid search for the best (side, price, volume) order."""
    prices = sorted(set([p for p, _ in bids] + [p for p, _ in asks]))
    lo, hi = min(prices) - 2, max(prices) + 5
    test_prices = list(range(lo, hi + 1))

    results = []
    for side in ("BUY", "SELL"):
        for price in test_prices:
            # Scan volumes at fine granularity
            for vol in range(500, max_vol + 1, 500):
                fill, clear = simulate(bids, asks, side, price, vol)
                if fill == 0:
                    continue
                if side == "BUY":
                    ppu = V - clear - buy_fee
                else:
                    ppu = clear - V - sell_fee
                pnl = fill * ppu
                results.append((pnl, side, price, vol, fill, clear, ppu))

    results.sort(key=lambda x: x[0], reverse=True)

    print(f"\n{'='*78}")
    print(f"  {name}  (V={V}, buy_fee={buy_fee}, sell_fee={sell_fee}, max_vol={max_vol})")
    print(f"{'='*78}")
    print(f"  {'Side':>4} {'Price':>6} {'Vol':>8} {'Fill':>8} {'Clear':>6} {'PPU':>7} {'PnL':>14}")
    print("  " + "-" * 74)
    seen = set()
    shown = 0
    for pnl, side, price, vol, fill, clear, ppu in results:
        key = (side, price, fill, clear)
        if key in seen:
            continue
        seen.add(key)
        print(f"  {side:>4} {price:>6} {vol:>8,} {fill:>8,} {clear:>6} {ppu:>7.2f} {pnl:>14,.2f}")
        shown += 1
        if shown >= 10:
            break

    if results:
        pnl, side, price, vol, fill, clear, ppu = results[0]
        return {"side": side, "price": price, "vol": vol, "fill": fill,
                "clear": clear, "ppu": ppu, "pnl": pnl}
    return None


if __name__ == "__main__":
    # ───────────────────────────────────────────────────
    # DRYLAND FLAX
    # ───────────────────────────────────────────────────
    flax_bids = [(30, 30000), (29, 5000), (28, 12000), (27, 28000)]
    flax_asks = [(28, 40000), (31, 20000), (32, 20000), (33, 30000)]
    best_flax = find_best(
        flax_bids, flax_asks,
        V=30, buy_fee=0, sell_fee=0,
        max_vol=30000, name="DRYLAND FLAX"
    )

    # ───────────────────────────────────────────────────
    # EMBER MUSHROOM
    # ───────────────────────────────────────────────────
    ember_bids = [(20, 43000), (19, 17000), (18, 6000), (17, 5000),
                  (16, 10000), (15, 5000), (14, 10000), (13, 7000)]
    ember_asks = [(12, 20000), (13, 25000), (14, 35000), (15, 6000),
                  (16, 5000), (18, 10000), (19, 12000)]
    best_ember = find_best(
        ember_bids, ember_asks,
        V=20, buy_fee=0.05, sell_fee=0.05,
        max_vol=43000, name="EMBER MUSHROOM"
    )

    # ───────────────────────────────────────────────────
    # FINAL SUBMISSION
    # ───────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  >>> FINAL SUBMISSION <<<")
    print("=" * 78)
    total = 0
    if best_flax:
        print(f"  DRYLAND FLAX:    {best_flax['side']} {best_flax['vol']:,} @ price {best_flax['price']}")
        print(f"                   -> fills {best_flax['fill']:,} @ clearing price {best_flax['clear']}, "
              f"profit {best_flax['pnl']:,.2f} XIRECs")
        total += best_flax['pnl']
    else:
        print("  DRYLAND FLAX:    No profitable order")
    if best_ember:
        print(f"  EMBER MUSHROOM:  {best_ember['side']} {best_ember['vol']:,} @ price {best_ember['price']}")
        print(f"                   -> fills {best_ember['fill']:,} @ clearing price {best_ember['clear']}, "
              f"profit {best_ember['pnl']:,.2f} XIRECs")
        total += best_ember['pnl']
    else:
        print("  EMBER MUSHROOM:  No profitable order")

    print(f"\n  TOTAL EXPECTED PROFIT: {total:,.2f} XIRECs")
    print("=" * 78)
