"""

======================================================================

--------------------------------
v5 quotes only 10 units per side with join_step=1. Observations from the
v5 run logs:
  • Fills ~8-9 units per side per tick when positioned near 0
  • Position swings in [-40, +40] range before rescue
  • Rescue (market-crossing at 35+) costs ~8 XIRECS per rescue unit
  • Quote size=10 caps the upside. With spread=16 and two-sided fills,
    each round-trip of 10 units = ~140 XIRECS. We need more per tick.

KEY IMPROVEMENTS IN THIS VERSION
--------------------------------
1. LARGER QUOTES: quote_size=30 (was 10). Our full limit=80 allows this.
   - On two-sided flow: 3× the round-trips → 3× the PnL
   - Position swings more, but priced-skew (below) prevents runaway

2. PRICE-BASED INVENTORY SKEW (critical):
   - v5 only skews quote *size*, not price. Heavy-long inventory?
     v5 still posts bid at best_bid+1, gets filled, position drifts.
   - v6 shifts both bid and ask DOWN when long (opposite when short).
   - Shift magnitude = 0.08 × position. At pos=+40 → shift = 3 ticks.
     → bid effectively abandons (won't fill), ask becomes aggressive
     → natural flattening without crossing the spread

3. TAKE PHASE RESTORED (with correct width):
   - v4 had take_width=2 which NEVER fires (spread=16).
   - v6 uses take_width based on HALF_SPREAD − 1 = 7. When a level
     appears ≤ 7 ticks from mid (rare but ~1-2% of ticks), cross.
   - Guaranteed fill at favorable price — worth the spread cost.

4. ADAPTIVE EDGE: if position is near zero, tighten edge to 1
   (grab more volume). If position heavy, widen to 3 (defensive).

5. PEPPER UNCHANGED: already optimal (~80k/day expected).
   Minor tweak: adverse_volume raised 7→8 to skip fewer good ticks.

EXPECTED BREAKDOWN (3 days):
  PEPPER:  ~240,000 XIRECS  (unchanged from v5)
  OSMIUM:  ~9,000-12,000 XIRECS  (was ~5,400)
  TOTAL:   ~249,000-252,000

Platform: position_limit=80, currency=XIRECS, conversions=1
"""

from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState,
)
from typing import Any, Dict, List, Optional, Tuple
import json
import jsonpickle


# ═══════════════════════════════════════════════════════════════════════════════
# Logger
# ═══════════════════════════════════════════════════════════════════════════════

class Logger:
    def __init__(self) -> None:
        self.logs: List[str] = []
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs.append(sep.join(str(o) for o in objects) + end)

    def flush(
        self,
        state:       TradingState,
        orders:      Dict[Symbol, List[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        base = [self.compress_state(state, ""), self.compress_orders(orders), conversions, "", ""]
        base_length = len(self.to_json(base))
        max_item    = (self.max_log_length - base_length) // 3
        logs_str    = "".join(self.logs)
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item),
            self.truncate(logs_str,    max_item),
        ]))
        self.logs = []

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[:max_length - 3] + "..."

    def compress_state(self, state: TradingState, trader_data: str) -> list:
        return [
            state.timestamp, trader_data,
            [[l.symbol, l.product] for l in state.listings.values()],
            {s: [od.buy_orders, od.sell_orders] for s, od in state.order_depths.items()},
            [[t.symbol, t.price, t.quantity] for ts in state.own_trades.values()    for t in ts],
            [[t.symbol, t.price, t.quantity] for ts in state.market_trades.values() for t in ts],
            state.position,
            [state.observations.plainValueObservations],
        ]

    def compress_orders(self, orders: Dict[Symbol, List[Order]]) -> list:
        return [[o.symbol, o.price, o.quantity] for os in orders.values() for o in os]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))


logger = Logger()


# ═══════════════════════════════════════════════════════════════════════════════
# Products & Parameters
# ═══════════════════════════════════════════════════════════════════════════════

class Product:
    PEPPER = "INTARIAN_PEPPER_ROOT"
    OSMIUM = "ASH_COATED_OSMIUM"


PARAMS: Dict[str, Any] = {

    Product.PEPPER: {
        "position_limit":   80,
        "adverse_volume":   8,       # Slightly relaxed vs v5's 7
        "prevent_adverse":  True,
        "impact_threshold": 0.05,
        "orderflow_weight": 0.3,
        "clear_min_edge":   3.0,
    },

    Product.OSMIUM: {
        "position_limit":   80,

        # ── EMA anchor ──────────────────────────────────────────────────
        "ema_alpha":        0.002,

        # ── Passive MAKE phase ──────────────────────────────────────────
        "edge":             2.0,     # Base edge from EMA for "resting level" scan
        "min_edge":         1.0,     # Tight edge when flat
        "max_edge":         3.0,     # Wide edge when heavy
        "edge_pos_scale":   40.0,    # |pos|=40 → shift from min_edge → max_edge
        "join_step":        1,       # Penny inside → become best bid/ask (first in queue)

        # ── Quote sizing ───────────────────────────────────────────────
        "quote_size":       35,      # 3× v5's 10 — exploits full capacity 30
        "min_quote_size":   5,       # Never drop below this on the "winning" side

        # ── Price-based inventory skew ──────────────────────────────────
        # Shift BOTH quotes by (skew_price × pos) in the direction that
        # reduces position. At pos=+40, shift = -3.2 → -3 ticks.
        "skew_price":       0.06, #0.08

        # ── Size-based inventory skew (multiplicative on the heavy side) ─
        "skew_size":        0.6,     # pos=+40 → sell-side qty ×1.3, buy-side ×0.7

        # ── TAKE phase (guaranteed fill on rare mispricing) ─────────────
        "take_enabled":     True,
        "take_width":       7.0,     # Cross if level within 7 ticks of fair
        "take_max_qty":     15,
        "take_adverse":     20,      # Skip if level volume > 20 (toxic)

        # ── Inventory rescue (cross to flatten) ────────────────────────
        "rescue_threshold": 50,      # Relaxed from 35 — price-skew does the work
        "rescue_qty":       8,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Trader
# ═══════════════════════════════════════════════════════════════════════════════

class Trader:
    """
    IMC Prosperity 4 Trader v6 — Deep-Quote + Take Edition.

    PEPPER : Aggressive buy at best_ask every tick → hold max +80
    OSMIUM : Queue-priority MM with large quotes + price-based skew + TAKE
    """

    def __init__(self, params: Dict = None):
        self.params = params or PARAMS
        self.LIMIT  = {p: self.params[p]["position_limit"] for p in self.params}

    # ─────────────────────────────────────────────────────────────────────────
    # Helper: filtered mid-price (adverse-volume filter)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _filtered_mid(
        od:         OrderDepth,
        adv_vol:    int,
        last_price: Optional[float],
    ) -> float:
        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)

        f_bids = [p for p in od.buy_orders  if od.buy_orders[p]       >= adv_vol]
        f_asks = [p for p in od.sell_orders if abs(od.sell_orders[p]) >= adv_vol]

        mm_bid = max(f_bids) if f_bids else None
        mm_ask = min(f_asks) if f_asks else None

        if mm_bid is None or mm_ask is None:
            return last_price if last_price is not None else (best_bid + best_ask) / 2.0
        return (mm_bid + mm_ask) / 2.0

    # ─────────────────────────────────────────────────────────────────────────
    # INTARIAN PEPPER ROOT — aggressive trend-long
    # ─────────────────────────────────────────────────────────────────────────

    def _pepper_fair(self, od: OrderDepth, obj: dict) -> Optional[float]:
        if not od.sell_orders or not od.buy_orders:
            return None

        p       = self.params[Product.PEPPER]
        last_px = obj.get("pepper_last_px")
        mid     = self._filtered_mid(od, p["adverse_volume"], last_px)
        obj["pepper_last_px"] = mid

        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        bv  = od.buy_orders[best_bid]
        av  = abs(od.sell_orders[best_ask])
        imb = (bv - av) / (bv + av + 1)

        return mid + p["orderflow_weight"] * imb

    def _pepper_orders(self, od: OrderDepth, fair: float, pos: int) -> List[Order]:
        orders: List[Order] = []
        p     = self.params[Product.PEPPER]
        limit = self.LIMIT[Product.PEPPER]

        # Aggressive BUY: lift best ask every tick
        if od.sell_orders and pos < limit:
            best_ask = min(od.sell_orders)
            ask_qty  = abs(od.sell_orders[best_ask])
            skip     = False

            if p["prevent_adverse"] and ask_qty > p["adverse_volume"]:
                skip = True

            if not skip and od.buy_orders:
                bq    = od.buy_orders[max(od.buy_orders)]
                total = ask_qty + bq
                if total > 0 and abs(ask_qty - bq) / total > p["impact_threshold"] and ask_qty > bq:
                    skip = True

            if not skip:
                qty = min(ask_qty, limit - pos)
                if qty > 0:
                    orders.append(Order(Product.PEPPER, best_ask, qty))

        # Light CLEAR: only trim if a bid is far above fair
        if od.buy_orders and pos > 0:
            best_bid = max(od.buy_orders)
            if best_bid >= fair + p["clear_min_edge"]:
                qty = min(od.buy_orders[best_bid], pos)
                if qty > 0:
                    orders.append(Order(Product.PEPPER, best_bid, -qty))

        return orders

    # ─────────────────────────────────────────────────────────────────────────
    # ASH COATED OSMIUM — TAKE + large-quote queue-priority MM
    # ─────────────────────────────────────────────────────────────────────────

    def _osmium_take(
        self,
        od:       OrderDepth,
        fair:     float,
        pos:      int,
    ) -> Tuple[List[Order], int, int]:
        """
        TAKE phase: cross spread on deeply mis-priced levels.
        Guaranteed fill (no queue). Skip toxic levels (volume > take_adverse).
        """
        orders: List[Order] = []
        p     = self.params[Product.OSMIUM]
        limit = self.LIMIT[Product.OSMIUM]

        if not p["take_enabled"]:
            return orders, 0, 0

        width    = p["take_width"]
        mqty     = p["take_max_qty"]
        adv      = p["take_adverse"]
        buy_vol  = 0
        sell_vol = 0

        # BUY: lift cheap asks
        if od.sell_orders:
            best_ask = min(od.sell_orders)
            ask_qty  = abs(od.sell_orders[best_ask])
            # cheap ask, non-toxic volume, room in position
            if best_ask <= fair - width and ask_qty <= adv:
                qty = min(ask_qty, mqty, limit - pos)
                if qty > 0:
                    orders.append(Order(Product.OSMIUM, best_ask, qty))
                    buy_vol += qty
                    # mutate od so subsequent phases see updated book
                    od.sell_orders[best_ask] += qty
                    if od.sell_orders[best_ask] == 0:
                        del od.sell_orders[best_ask]

        # SELL: hit rich bids
        if od.buy_orders:
            best_bid = max(od.buy_orders)
            bid_qty  = od.buy_orders[best_bid]
            if best_bid >= fair + width and bid_qty <= adv:
                qty = min(bid_qty, mqty, limit + pos)
                if qty > 0:
                    orders.append(Order(Product.OSMIUM, best_bid, -qty))
                    sell_vol += qty
                    od.buy_orders[best_bid] -= qty
                    if od.buy_orders[best_bid] == 0:
                        del od.buy_orders[best_bid]

        return orders, buy_vol, sell_vol

    def _osmium_make(
        self,
        od:       OrderDepth,
        pos:      int,
        buy_vol:  int,
        sell_vol: int,
        obj:      dict,
    ) -> List[Order]:
        """
        MAKE phase: large passive quotes with queue priority + price skew.

        1. EMA-anchored fair value.
        2. Adaptive edge: tight when flat, wide when heavy.
        3. Join: 1 tick inside nearest resting level → first-in-queue.
        4. PRICE SKEW: shift both quotes toward reducing inventory.
        5. SIZE SKEW: bigger on the reducing side, smaller on the building side.
        """
        orders: List[Order] = []
        p     = self.params[Product.OSMIUM]
        limit = self.LIMIT[Product.OSMIUM]

        if not od.sell_orders or not od.buy_orders:
            return orders

        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        mid      = (best_bid + best_ask) / 2.0

        # Update EMA (slow anchor)
        prev_ema = obj.get("osmium_ema", mid)
        ema      = p["ema_alpha"] * mid + (1.0 - p["ema_alpha"]) * prev_ema
        obj["osmium_ema"] = ema

        # ── Inventory RESCUE (aggressive, cross spread to flatten) ────────
        pos_eff = pos + buy_vol - sell_vol
        if pos_eff < -p["rescue_threshold"]:
            qty = min(p["rescue_qty"], limit - pos_eff)
            if qty > 0:
                orders.append(Order(Product.OSMIUM, best_ask, qty))
            return orders

        if pos_eff > p["rescue_threshold"]:
            qty = min(p["rescue_qty"], limit + pos_eff)
            if qty > 0:
                orders.append(Order(Product.OSMIUM, best_bid, -qty))
            return orders

        # ── Adaptive edge: tighter when flat, wider when heavy ────────────
        pos_frac = min(1.0, abs(pos_eff) / p["edge_pos_scale"])
        edge = p["min_edge"] + (p["max_edge"] - p["min_edge"]) * pos_frac

        # ── Find resting levels outside EMA±edge ──────────────────────────
        step       = p["join_step"]
        asks_above = [px for px in od.sell_orders if px > ema + edge]
        bids_below = [px for px in od.buy_orders  if px < ema - edge]

        if asks_above and bids_below:
            our_ask = min(asks_above) - step
            our_bid = max(bids_below) + step
        else:
            # Fallback: penny inside raw best bid/ask
            our_ask = best_ask - step
            our_bid = best_bid + step

        # ── PRICE SKEW: shift quotes to reduce inventory ──────────────────
        # Long  → shift DOWN (bid disengages, ask becomes aggressive)
        # Short → shift UP   (ask disengages, bid becomes aggressive)
        price_shift = int(round(-p["skew_price"] * pos_eff))
        our_ask += price_shift
        our_bid += price_shift

        # Safety: ensure quotes don't cross the other side of the book
        if our_ask <= best_bid:
            our_ask = best_bid + 1
        if our_bid >= best_ask:
            our_bid = best_ask - 1

        # Safety: ensure our own quotes don't cross
        if our_bid >= our_ask:
            return orders

        # ── SIZE SKEW: bigger on reducing side, smaller on building side ──
        pos_norm = pos_eff / limit  # ∈ [-1, +1]
        skew_f   = p["skew_size"]
        qsize    = p["quote_size"]
        min_q    = p["min_quote_size"]

        # pos > 0 → want to sell → buy_q smaller, sell_q larger
        buy_q  = int(qsize * (1.0 - skew_f * pos_norm))
        sell_q = int(qsize * (1.0 + skew_f * pos_norm))
        buy_q  = max(min_q, buy_q)
        sell_q = max(min_q, sell_q)

        # Respect position limits
        buy_q  = min(buy_q,  limit - (pos + buy_vol))
        sell_q = min(sell_q, limit + (pos - sell_vol))

        if buy_q > 0:
            orders.append(Order(Product.OSMIUM, our_bid, buy_q))
        if sell_q > 0:
            orders.append(Order(Product.OSMIUM, our_ask, -sell_q))

        return orders

    def _osmium_fair(self, od: OrderDepth, obj: dict) -> Optional[float]:
        """Fair value = filtered mid, used for TAKE-phase decisions."""
        if not od.sell_orders or not od.buy_orders:
            return None
        # Use adverse_volume=9 as in v4 for filtered-mid (OSMIUM-specific)
        last_px = obj.get("osmium_last_px")
        mid     = self._filtered_mid(od, 9, last_px)
        obj["osmium_last_px"] = mid
        return mid

    # ─────────────────────────────────────────────────────────────────────────
    # run()
    # ─────────────────────────────────────────────────────────────────────────

    def run(
        self, state: TradingState
    ) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        obj: dict = {}
        if state.traderData:
            try:
                obj = jsonpickle.decode(state.traderData)
            except Exception:
                obj = {}

        result: Dict[Symbol, List[Order]] = {}

        # ── INTARIAN PEPPER ROOT ──────────────────────────────────────────
        if Product.PEPPER in state.order_depths:
            od   = state.order_depths[Product.PEPPER]
            pos  = state.position.get(Product.PEPPER, 0)
            fair = self._pepper_fair(od, obj)
            result[Product.PEPPER] = (
                self._pepper_orders(od, fair, pos) if fair is not None else []
            )

        # ── ASH COATED OSMIUM: TAKE → MAKE ────────────────────────────────
        if Product.OSMIUM in state.order_depths:
            od   = state.order_depths[Product.OSMIUM]
            pos  = state.position.get(Product.OSMIUM, 0)
            fair = self._osmium_fair(od, obj)

            if fair is None:
                result[Product.OSMIUM] = []
            else:
                t_orders, bv, sv = self._osmium_take(od, fair, pos)
                m_orders         = self._osmium_make(od, pos, bv, sv, obj)
                result[Product.OSMIUM] = t_orders + m_orders

        trader_data = jsonpickle.encode(obj)
        conversions = 1

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
