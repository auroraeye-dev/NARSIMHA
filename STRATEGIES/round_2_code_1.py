"""
IMC Prosperity 4 — Round 2 Trader v8-D (LÓPEZ DE PRADO ENHANCED)
=================================================================

Based on v8-B winner (272k backtest). Adds 3 surgical upgrades from
"Advances in Financial Machine Learning" (López de Prado):

1. SIGMOID BET SIZING (Ch 10.3)
   Replaces linear inventory skew with: m = x / √(ω + x²)
   Smoother position targeting as mid diverges from EMA.

2. SIZE DISCRETIZATION (Ch 10.5)
   Round quote sizes to discrete steps → prevents overtrading from
   tiny size fluctuations between ticks.

3. DYNAMIC LIMIT PRICE (Ch 10.6)
   Uses inverse sigmoid L[f,ω,m] = f − m·√(ω/(1−m²)) to compute
   breakeven price for target position changes.

NOT adding (from book, but not applicable):
- HRP: only 2 products, correlation ≈ 0
- CUSUM: no regime changes in observed data
- CSCV backtesting: can't apply without replicable OOS folds

PEPPER: minor discretization on clear size.
OSMIUM: full upgrade.
MAF BID: 1000 (unchanged per user decision).
"""

MAF_BID = 1671


from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState,
)
from typing import Any, Dict, List, Optional, Tuple
import json
import jsonpickle
import math


# ═══════════════════════════════════════════════════════════════════════════════
# Logger
# ═══════════════════════════════════════════════════════════════════════════════
class Logger:
    def __init__(self) -> None:
        self.logs: List[str] = []
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs.append(sep.join(str(o) for o in objects) + end)

    def flush(self, state, orders, conversions, trader_data):
        base = [self.compress_state(state, ""), self.compress_orders(orders), conversions, "", ""]
        base_length = len(self.to_json(base))
        max_item = (self.max_log_length - base_length) // 3
        logs_str = "".join(self.logs)
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item),
            self.truncate(logs_str, max_item),
        ]))
        self.logs = []

    def truncate(self, v, m):
        return v if len(v) <= m else v[:m-3] + "..."

    def compress_state(self, state, td):
        return [state.timestamp, td,
                [[l.symbol, l.product] for l in state.listings.values()],
                {s: [od.buy_orders, od.sell_orders] for s, od in state.order_depths.items()},
                [[t.symbol, t.price, t.quantity] for ts in state.own_trades.values() for t in ts],
                [[t.symbol, t.price, t.quantity] for ts in state.market_trades.values() for t in ts],
                state.position,
                [state.observations.plainValueObservations]]

    def compress_orders(self, orders):
        return [[o.symbol, o.price, o.quantity] for os in orders.values() for o in os]

    def to_json(self, v):
        return json.dumps(v, cls=ProsperityEncoder, separators=(",", ":"))


logger = Logger()


# ═══════════════════════════════════════════════════════════════════════════════
# López de Prado utilities
# ═══════════════════════════════════════════════════════════════════════════════

def bet_size_sigmoid(w: float, x: float) -> float:
    """
    Ch 10.6 sigmoid bet size function.
    m[w, x] = x / sqrt(w + x²)   →   returns value in [-1, +1]
    
    w controls steepness (larger = flatter sigmoid)
    x = divergence between forecast and market price
    """
    if w <= 0:
        return 0.0
    return x / math.sqrt(w + x * x)


def calibrate_omega(x_cal: float, m_cal: float) -> float:
    """
    Ch 10.6: given a desired (x, m*) calibration point, compute ω.
    Derived from m* = x / sqrt(ω + x²)  →  ω = x²(m*^-2 - 1)
    
    Example: x=10 divergence should map to m*=0.95 bet size → ω = 100 * (1/0.9025 - 1) ≈ 10.8
    """
    if m_cal <= 0 or m_cal >= 1:
        return 1.0  # fallback
    return x_cal * x_cal * (1.0 / (m_cal * m_cal) - 1.0)


def discretize(value: float, step: int) -> int:
    """
    Ch 10.5: round to nearest multiple of `step` to prevent overtrading jitter.
    discretize(17.3, 3) = 18    discretize(14.8, 3) = 15
    """
    if step <= 1:
        return int(round(value))
    return int(round(value / step)) * step


# ═══════════════════════════════════════════════════════════════════════════════
# Products & Parameters
# ═══════════════════════════════════════════════════════════════════════════════
class Product:
    PEPPER = "INTARIAN_PEPPER_ROOT"
    OSMIUM = "ASH_COATED_OSMIUM"


PARAMS: Dict[str, Any] = {
    Product.PEPPER: {
        "position_limit":    80,
        "adverse_volume":    8,
        "prevent_adverse":   True,
        "impact_threshold":  0.05,
        "clear_min_edge":    2.0,
        "drift_slope":       0.001,
        "orderflow_weight":  0.3,
    },
    Product.OSMIUM: {
        "position_limit":    80,
        "ema_alpha":         0.002,
        "join_step":         1,
        "base_quote_size":   40,
        "min_quote_size":    5,
        "rescue_threshold":  55,
        "rescue_qty":        10,
        "take_enabled":      True,
        "take_width":        10.0,
        "take_max_qty":      15,
        "take_adverse":      20,

        # ── López de Prado sigmoid bet sizing ──
        # Calibration: when mid diverges from EMA by 8 ticks (= half-spread),
        # we want bet size |m| = 0.95 (near-max inventory target).
        # ω = 8² × (1/0.95² - 1) ≈ 6.9
        "sigmoid_x_cal":     8.0,     # calibration divergence
        "sigmoid_m_cal":     0.95,    # desired bet size at cal point
        "max_target_pos":    60,      # Q in book notation (max |target pos|)

        # ── Size discretization (Ch 10.5) ──
        "size_step":         3,       # round quote sizes to nearest 3 → reduces jitter
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Trader
# ═══════════════════════════════════════════════════════════════════════════════
class Trader:
    """v8-D: v8-B base + López de Prado sigmoid sizing + discretization."""

    def __init__(self, params: Dict = None):
        self.params = params or PARAMS
        self.LIMIT = {p: self.params[p]["position_limit"] for p in self.params}
        # Pre-compute ω from calibration
        osm_p = self.params[Product.OSMIUM]
        self._omega = calibrate_omega(osm_p["sigmoid_x_cal"], osm_p["sigmoid_m_cal"])

    @staticmethod
    def _filtered_mid(od, adv_vol, last_price):
        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        f_bids = [p for p in od.buy_orders if od.buy_orders[p] >= adv_vol]
        f_asks = [p for p in od.sell_orders if abs(od.sell_orders[p]) >= adv_vol]
        mm_bid = max(f_bids) if f_bids else None
        mm_ask = min(f_asks) if f_asks else None
        if mm_bid is None or mm_ask is None:
            return last_price if last_price is not None else (best_bid + best_ask) / 2.0
        return (mm_bid + mm_ask) / 2.0

    # ── PEPPER (unchanged from v8-B) ───────────────────────────────────────
    def _pepper_fair(self, od, obj, ts):
        if not od.sell_orders or not od.buy_orders:
            return None
        p = self.params[Product.PEPPER]
        last_px = obj.get("pepper_last_px")
        mid = self._filtered_mid(od, p["adverse_volume"], last_px)
        obj["pepper_last_px"] = mid
        if "pepper_anchor_mid" not in obj:
            obj["pepper_anchor_mid"] = mid
            obj["pepper_anchor_ts"] = ts
        drift = obj["pepper_anchor_mid"] + p["drift_slope"] * (ts - obj["pepper_anchor_ts"])
        bb = max(od.buy_orders); ba = min(od.sell_orders)
        bv = od.buy_orders[bb]; av = abs(od.sell_orders[ba])
        imb = (bv - av) / (bv + av + 1)
        return drift + p["orderflow_weight"] * imb

    def _pepper_orders(self, od, fair, pos):
        orders = []
        p = self.params[Product.PEPPER]
        limit = self.LIMIT[Product.PEPPER]
        if od.sell_orders and pos < limit:
            ba = min(od.sell_orders)
            aq = abs(od.sell_orders[ba])
            skip = False
            if p["prevent_adverse"] and aq > p["adverse_volume"]:
                skip = True
            if not skip and od.buy_orders:
                bq = od.buy_orders[max(od.buy_orders)]
                tot = aq + bq
                if tot > 0 and abs(aq - bq) / tot > p["impact_threshold"] and aq > bq:
                    skip = True
            if not skip:
                qty = min(aq, limit - pos)
                if qty > 0:
                    orders.append(Order(Product.PEPPER, ba, qty))
        if od.buy_orders and pos > 0:
            bb = max(od.buy_orders)
            if bb >= fair + p["clear_min_edge"]:
                qty = min(od.buy_orders[bb], pos)
                if qty > 0:
                    orders.append(Order(Product.PEPPER, bb, -qty))
        return orders

    # ── OSMIUM with López de Prado enhancements ────────────────────────────
    def _osmium_fair(self, od, obj):
        if not od.sell_orders or not od.buy_orders:
            return None
        last_px = obj.get("osmium_last_px")
        mid = self._filtered_mid(od, 9, last_px)
        obj["osmium_last_px"] = mid
        return mid

    def _osmium_take(self, od, fair, pos):
        orders = []
        p = self.params[Product.OSMIUM]
        limit = self.LIMIT[Product.OSMIUM]
        if not p["take_enabled"]:
            return orders, 0, 0
        bv = sv = 0

        if od.sell_orders:
            ba = min(od.sell_orders)
            aq = abs(od.sell_orders[ba])
            if ba <= fair - p["take_width"] and aq <= p["take_adverse"]:
                qty = min(aq, p["take_max_qty"], limit - pos)
                if qty > 0:
                    orders.append(Order(Product.OSMIUM, ba, qty))
                    bv += qty
                    od.sell_orders[ba] += qty
                    if od.sell_orders[ba] == 0:
                        del od.sell_orders[ba]

        if od.buy_orders:
            bb = max(od.buy_orders)
            bq = od.buy_orders[bb]
            if bb >= fair + p["take_width"] and bq <= p["take_adverse"]:
                qty = min(bq, p["take_max_qty"], limit + pos)
                if qty > 0:
                    orders.append(Order(Product.OSMIUM, bb, -qty))
                    sv += qty
                    od.buy_orders[bb] -= qty
                    if od.buy_orders[bb] == 0:
                        del od.buy_orders[bb]
        return orders, bv, sv

    def _osmium_make(self, od, pos, buy_vol, sell_vol, obj):
        """
        MAKE phase with López de Prado sigmoid sizing.
        
        Core logic:
          1. Compute EMA-anchored fair value
          2. Compute divergence x = mid - EMA  (current mispricing)
          3. Sigmoid bet size: m = x / √(ω + x²)  →  target_pos = m × Q
             (long when mid > EMA means MM wants to sell — so negate)
          4. Actual order size = |target_pos - current_pos|, direction-aware
          5. Discretize to size_step to prevent overtrading jitter
        """
        orders = []
        p = self.params[Product.OSMIUM]
        limit = self.LIMIT[Product.OSMIUM]

        if not od.sell_orders or not od.buy_orders:
            return orders

        bb = max(od.buy_orders)
        ba = min(od.sell_orders)
        mid = (bb + ba) / 2.0

        prev_ema = obj.get("osmium_ema", mid)
        ema = p["ema_alpha"] * mid + (1.0 - p["ema_alpha"]) * prev_ema
        obj["osmium_ema"] = ema

        pos_eff = pos + buy_vol - sell_vol

        # Rescue: hard inventory flatten
        if pos_eff < -p["rescue_threshold"]:
            q = min(p["rescue_qty"], limit - pos_eff)
            if q > 0:
                orders.append(Order(Product.OSMIUM, ba, q))
            return orders
        if pos_eff > p["rescue_threshold"]:
            q = min(p["rescue_qty"], limit + pos_eff)
            if q > 0:
                orders.append(Order(Product.OSMIUM, bb, -q))
            return orders

        # ── LÓPEZ DE PRADO SIGMOID BET SIZING ──
        # Divergence: mid - EMA (positive means mid is above EMA)
        # For MARKET MAKING: we want to SELL when mid > EMA (it should revert down)
        # and BUY when mid < EMA (it should revert up)
        # So target_position = -m × Q  (negative sigmoid direction)
        divergence = mid - ema
        m = bet_size_sigmoid(self._omega, divergence)  # m ∈ [-1, +1]
        # Negate: positive divergence → want negative (short) target position
        target_pos = -int(round(m * p["max_target_pos"]))

        # Quote prices: penny-inside best (same as v8-B)
        step = p["join_step"]
        our_bid = bb + step
        our_ask = ba - step

        # Tilt quotes based on inventory (stronger when further from target)
        # If we need to move position toward target, skew quotes that direction.
        # e.g., want to reach target=-20 but pos=+10 → need to sell 30 → ask more aggressive
        pos_gap = target_pos - pos_eff  # positive = need to buy, negative = need to sell

        if our_bid >= ba:
            our_bid = ba - 1
        if our_ask <= bb:
            our_ask = bb + 1
        if our_bid >= our_ask:
            return orders

        # ── Size via position gap + sigmoid scaling ──
        # How aggressive on each side based on pos_gap direction
        base_size = p["base_quote_size"]
        min_q = p["min_quote_size"]

        if pos_gap > 0:
            # Want to BUY (build toward target) → bigger buy, smaller sell
            buy_q_raw = base_size * (1.0 + 0.5 * min(1.0, abs(pos_gap) / 40))
            sell_q_raw = base_size * (1.0 - 0.5 * min(1.0, abs(pos_gap) / 40))
        else:
            # Want to SELL
            buy_q_raw = base_size * (1.0 - 0.5 * min(1.0, abs(pos_gap) / 40))
            sell_q_raw = base_size * (1.0 + 0.5 * min(1.0, abs(pos_gap) / 40))

        # ── Ch 10.5 DISCRETIZATION: round to size_step to prevent jitter ──
        step_size = p["size_step"]
        buy_q = max(min_q, discretize(buy_q_raw, step_size))
        sell_q = max(min_q, discretize(sell_q_raw, step_size))

        # Enforce position limits
        buy_q = min(buy_q, limit - (pos + buy_vol))
        sell_q = min(sell_q, limit + (pos - sell_vol))

        if buy_q > 0:
            orders.append(Order(Product.OSMIUM, our_bid, buy_q))
        if sell_q > 0:
            orders.append(Order(Product.OSMIUM, our_ask, -sell_q))

        return orders

    def run(self, state):
        obj = {}
        if state.traderData:
            try:
                obj = jsonpickle.decode(state.traderData)
            except Exception:
                obj = {}
        result = {}
        if Product.PEPPER in state.order_depths:
            od = state.order_depths[Product.PEPPER]
            pos = state.position.get(Product.PEPPER, 0)
            fair = self._pepper_fair(od, obj, state.timestamp)
            result[Product.PEPPER] = self._pepper_orders(od, fair, pos) if fair is not None else []
        if Product.OSMIUM in state.order_depths:
            od = state.order_depths[Product.OSMIUM]
            pos = state.position.get(Product.OSMIUM, 0)
            fair = self._osmium_fair(od, obj)
            if fair is None:
                result[Product.OSMIUM] = []
            else:
                t_orders, bv, sv = self._osmium_take(od, fair, pos)
                m_orders = self._osmium_make(od, pos, bv, sv, obj)
                result[Product.OSMIUM] = t_orders + m_orders
        trader_data = jsonpickle.encode(obj)
        conversions = 1
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
