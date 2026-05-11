"""
IMC Prosperity 4 - Round 3: Bio-Pod Solver v3
HINT INTEGRATED: reserves are on a discrete grid of multiples of 5
("the power of flowering fives, always set a flowering five apart")

This changes everything:
  - Reserves live on {670, 675, 680, ..., 915, 920} — 51 discrete values
  - Any bid not a multiple of 5 wastes money (clears same pods as nearest lower multiple of 5)
  - The optimal bid space is itself the grid {670, 675, ..., 920}

We test THREE reserve distributions on the 5-grid:
  A) Uniform over all 51 grid points
  B) Two-band split: low band at {670,675,...,710}, high band at {810,815,...,920}
  C) "Power of fives" — heavier weight on multiples of 25 or 50
"""

import numpy as np
from collections import Counter

V = 920
GRID = list(range(670, 921, 5))   # [670, 675, ..., 920], 51 values


# ============================================================
#  DISCRETE RESERVE DISTRIBUTIONS
#  Each returns a dict {reserve_value: probability_mass}
# ============================================================

def dist_uniform():
    """Each grid point equally likely."""
    p = 1.0 / len(GRID)
    return {r: p for r in GRID}

def dist_twoband():
    """Reserves in {670..710} or {810..920}, 50/50 split within each."""
    low = [r for r in GRID if 670 <= r <= 710]   # 9 values
    high = [r for r in GRID if 810 <= r <= 920]  # 23 values
    d = {}
    for r in low:  d[r] = 0.5 / len(low)
    for r in high: d[r] = 0.5 / len(high)
    return d

def dist_power_fives():
    """'Flowering fives' with extra weight on multiples of 50, then 25, then 5."""
    d = {}
    for r in GRID:
        if r % 50 == 0: d[r] = 4.0
        elif r % 25 == 0: d[r] = 2.0
        else: d[r] = 1.0
    total = sum(d.values())
    return {r: w / total for r, w in d.items()}

def dist_power_fives_bands():
    """Combine the two-band hypothesis with power-of-fives weighting."""
    base = dist_twoband()
    d = {}
    for r, p in base.items():
        if r % 50 == 0: d[r] = p * 4.0
        elif r % 25 == 0: d[r] = p * 2.0
        else: d[r] = p * 1.0
    total = sum(d.values())
    return {r: w / total for r, w in d.items()}

RESERVE_DISTS = {
    "uniform_on_fives":       dist_uniform(),
    "twoband_on_fives":       dist_twoband(),
    "power_of_fives":         dist_power_fives(),
    "power_of_fives_banded":  dist_power_fives_bands(),
}


# ============================================================
#  EXPECTED PROFIT (DISCRETE VERSION)
# ============================================================

def expected_profit_discrete(b1, b2, avg_bid2, reserve_dist):
    """
    Expected profit per Guardener when reserves are discrete.
    Sum over each reserve value r: P(r) * profit_given_reserve(r).
    """
    if b2 < b1 or b1 < 670 or b2 > 920:
        return -1e9

    # Penalty scale on bid 2
    if b2 >= avg_bid2:
        scale = 1.0
    else:
        scale = ((V - avg_bid2) / (V - b2)) ** 3 if b2 < V else 0.0

    total = 0.0
    for r, p in reserve_dist.items():
        if b1 >= r:
            # bid 1 wins — margin V - b1
            total += p * (V - b1)
        elif b2 >= r:
            # bid 2 wins — margin (V - b2) * scale
            total += p * (V - b2) * scale
        # else: no trade
    return total


# ============================================================
#  GRID SEARCH — RESTRICTED TO MULTIPLES OF 5
# ============================================================

def grid_search_fives(objective_fn):
    """Bids are also on the 5-grid, since any other bid is wasteful."""
    best = (-1e18, None)
    for b1 in GRID:
        for b2 in GRID:
            if b2 < b1: continue
            v = objective_fn(b1, b2)
            if v > best[0]:
                best = (v, (b1, b2))
    return best


# ============================================================
#  MAIN: run everything
# ============================================================

def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


if __name__ == "__main__":

    section("0. RESERVE DISTRIBUTIONS ON THE 5-GRID — sanity check")
    for name, d in RESERVE_DISTS.items():
        sample = {k: round(v, 4) for k, v in list(d.items())[:6]}
        print(f"  {name:<25} total mass = {sum(d.values()):.3f}, sample: {sample}...")

    # --------------------------------------------------------
    section("1. OPTIMAL BIDS UNDER EACH DISCRETE MODEL (avg = 870)")
    for name, dist in RESERVE_DISTS.items():
        val, bids = grid_search_fives(lambda b1, b2: expected_profit_discrete(b1, b2, 870, dist))
        print(f"  {name:<25}: bids={bids}, EV/pod = {val:.3f}")

    # --------------------------------------------------------
    section("2. OPTIMAL BIDS UNDER DIFFERENT CROWD-AVG SCENARIOS")
    print("  For each model, shows how optimal bid 2 shifts with the crowd average\n")
    for avg in [860, 865, 870, 872, 875, 880, 885]:
        print(f"  avg={avg}:")
        for name, dist in RESERVE_DISTS.items():
            val, bids = grid_search_fives(lambda b1, b2: expected_profit_discrete(b1, b2, avg, dist))
            print(f"    {name:<25}: bids={bids}, EV/pod = {val:.3f}")

    # --------------------------------------------------------
    section("3. BAYESIAN BEST BID — prior over crowd avg")
    print("  Prior: discrete, P(avg) concentrated on multiples of 5 near 870\n")
    # realistic prior: most mass on 870, some on 865 and 875, tails at 860/880
    crowd_prior = {860: 0.05, 865: 0.20, 870: 0.40, 875: 0.25, 880: 0.08, 885: 0.02}

    def bayes_profit(b1, b2, dist):
        return sum(w * expected_profit_discrete(b1, b2, a, dist) for a, w in crowd_prior.items())

    for name, dist in RESERVE_DISTS.items():
        val, bids = grid_search_fives(lambda b1, b2: bayes_profit(b1, b2, dist))
        print(f"  {name:<25}: bids={bids}, Bayesian EV/pod = {val:.3f}")

    # --------------------------------------------------------
    section("4. MODEL-AVERAGED BAYESIAN SOLUTION")
    print("  Weight reserve models: 40% twoband, 30% power_fives_banded, 20% power_fives, 10% uniform\n")
    model_weights = {
        "twoband_on_fives":       0.40,
        "power_of_fives_banded":  0.30,
        "power_of_fives":         0.20,
        "uniform_on_fives":       0.10,
    }

    def model_avg_profit(b1, b2):
        return sum(w * bayes_profit(b1, b2, RESERVE_DISTS[m]) for m, w in model_weights.items())

    val, bids = grid_search_fives(model_avg_profit)
    print(f"  MODEL-AVERAGED OPTIMUM: bids={bids}, EV/pod = {val:.3f}")

    # --------------------------------------------------------
    section("5. HEAD-TO-HEAD: candidate bid pairs, all models, Bayesian avg prior")
    # All candidates are multiples of 5
    candidates = [(710, 865), (710, 870), (710, 875), (710, 880),
                  (715, 870), (720, 870), (770, 870), (795, 870)]

    header = f"  {'bids':<12}"
    for m in model_weights: header += f"{m[:20]:<22}"
    header += "model-avg"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for bids in candidates:
        row = f"  {str(bids):<12}"
        model_ev = 0.0
        for m, w in model_weights.items():
            ev = bayes_profit(bids[0], bids[1], RESERVE_DISTS[m])
            row += f"{ev:<22.3f}"
            model_ev += w * ev
        row += f"{model_ev:.3f}"
        print(row)

    # --------------------------------------------------------
    section("6. THE FINAL ANSWER")
    best_val, best_bids = grid_search_fives(model_avg_profit)
    print(f"\n  Recommended bid:      {best_bids}")
    print(f"  Expected profit/pod:  {best_val:.3f}")
    print(f"  Both values are multiples of 5 ('flowering fives').")

    print("\n  How many Guardeners you'd sweep (in the main twoband model):")
    d = RESERVE_DISTS["twoband_on_fives"]
    b1, b2 = best_bids
    sweep_by_b1 = sum(p for r, p in d.items() if b1 >= r)
    sweep_by_b2 = sum(p for r, p in d.items() if b2 >= r and b1 < r)
    print(f"    Fraction swept by bid 1 ({b1}): {sweep_by_b1:.3f}")
    print(f"    Fraction swept by bid 2 ({b2}): {sweep_by_b2:.3f}")
    print(f"    Not swept: {1 - sweep_by_b1 - sweep_by_b2:.3f}")
