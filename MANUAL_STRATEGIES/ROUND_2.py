"""
Prosperity 4 — Round 2 Manual Challenge: 3-Pillar Investment Optimizer

KNOWN:
  - 3 pillars: Research, Scale, Speed (sum = 100%)

import numpy as np
from scipy.optimize import minimize

BUDGET = 50000

# ═══════════════════════════════════════════════════════════════════════════════
# REVERSE-ENGINEER RESEARCH FORMULA
# ═══════════════════════════════════════════════════════════════════════════════
# Research: 11,000 → 1,35,879
# Ratio: 135879 / 11000 = 12.35 
# 
# Common functional forms to test:
#   Linear:      R_out = a × R_in                 → a = 12.35
#   Sqrt:        R_out = a × sqrt(R_in)           → a = 1295
#   Log:         R_out = a × log(R_in + 1)        → a = 14601
#   Power 0.75:  R_out = a × R_in^0.75            → a = 125.6
#   Power 1.25:  R_out = a × R_in^1.25            → a = 1.216

R_in = 11000
R_out = 135879

print("=" * 70)
print("RESEARCH FORMULA CANDIDATES (reverse from 11,000 → 135,879)")
print("=" * 70)
forms = {
    "Linear (R × a)":            ("linear", R_out / R_in),
    "Sqrt (sqrt(R) × a)":        ("sqrt",   R_out / np.sqrt(R_in)),
    "Log (log(R) × a)":          ("log",    R_out / np.log(R_in + 1)),
    "Power 0.5 (R^0.5 × a)":     ("pow0.5", R_out / R_in**0.5),
    "Power 0.75 (R^0.75 × a)":   ("pow0.75",R_out / R_in**0.75),
    "Power 1.25 (R^1.25 × a)":   ("pow1.25",R_out / R_in**1.25),
}
for name, (form, coef) in forms.items():
    print(f"  {name:<30} → coef = {coef:.4f}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# REVERSE-ENGINEER SCALE MULTIPLIER
# ═══════════════════════════════════════════════════════════════════════════════
# Scale: 37,500 → 5.3x
# Possible forms:
#   Linear:      S_mult = 1 + b × S_in             → b ≈ 0.0001147
#   Sqrt:        S_mult = 1 + b × sqrt(S_in)       → b ≈ 0.0222
#   Log:         S_mult = 1 + b × log(S_in + 1)    → b ≈ 0.4096
#   Power 0.5:   S_mult = 1 + b × S_in^0.5         → b ≈ 0.0222

S_in = 37500
S_mult = 5.3

print("=" * 70)
print("SCALE FORMULA CANDIDATES (reverse from 37,500 → 5.3x)")
print("=" * 70)
scale_forms = {
    "Linear (1 + b × S)":         ("linear", (S_mult - 1) / S_in),
    "Sqrt (1 + b × sqrt(S))":     ("sqrt",   (S_mult - 1) / np.sqrt(S_in)),
    "Log (1 + b × log(S))":       ("log",    (S_mult - 1) / np.log(S_in + 1)),
    "Power 0.5":                  ("pow0.5", (S_mult - 1) / S_in**0.5),
    "Power 0.75":                 ("pow0.75",(S_mult - 1) / S_in**0.75),
}
for name, (form, coef) in scale_forms.items():
    print(f"  {name:<30} → coef = {coef:.6f}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# MOST LIKELY MODEL: Sqrt forms (standard for investment games)
# Diminishing returns is realistic for both pillars
# ═══════════════════════════════════════════════════════════════════════════════

# Best guess formulas (sqrt diminishing returns)
def research_output(R):
    """R seashells → base value. Using sqrt(R) × 1295 to match 11K → 135,879."""
    if R <= 0: return 0
    return 1295 * np.sqrt(R)

def scale_mult(S):
    """S seashells → multiplier. Using 1 + 0.0222 × sqrt(S) to match 37.5K → 5.3x."""
    if S <= 0: return 1.0
    return 1.0 + 0.0222 * np.sqrt(S)

def speed_mult(Sp):
    """Speed unknown — ASSUME similar shape, starts at 1.0.
       Test multiple strengths later."""
    if Sp <= 0: return 1.0
    return 1.0 + 0.02 * np.sqrt(Sp)   # conservative guess, tune below

# Verify current allocation matches known forecasts
R_chk = research_output(11000)
S_chk = scale_mult(37500)
print("=" * 70)
print("MODEL VERIFICATION with current split 22/75/3")
print("=" * 70)
print(f"  Research(11,000) predicted : {R_chk:,.0f}   (actual: 135,879)  ✓ match if close")
print(f"  Scale(37,500)    predicted : {S_chk:.3f}x   (actual: 5.3x)      ✓ match if close")
print(f"  Speed(1,500)     assumed   : {speed_mult(1500):.3f}x   (actual: unknown)")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION: find R%, S%, Sp% that maximize payoff
# ═══════════════════════════════════════════════════════════════════════════════
def total_payoff(alloc):
    """alloc = [R_pct, S_pct, Sp_pct] all in 0-100 range, must sum to 100."""
    R_pct, S_pct, Sp_pct = alloc
    R_amt = BUDGET * R_pct / 100
    S_amt = BUDGET * S_pct / 100
    Sp_amt = BUDGET * Sp_pct / 100
    return research_output(R_amt) * scale_mult(S_amt) * speed_mult(Sp_amt)

# Evaluate current split
current = [22, 75, 3]
current_payoff = total_payoff(current)
print(f"Current split 22/75/3 → estimated payoff: {current_payoff:,.0f}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# GRID SEARCH over all valid allocations
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("GRID SEARCH: optimal allocation")
print("=" * 70)

best_payoff = 0
best_alloc = None
for r in range(1, 99):
    for s in range(1, 100 - r):
        sp = 100 - r - s
        if sp < 1: continue
        payoff = total_payoff([r, s, sp])
        if payoff > best_payoff:
            best_payoff = payoff
            best_alloc = [r, s, sp]

print(f"OPTIMAL SPLIT: Research {best_alloc[0]}% / Scale {best_alloc[1]}% / Speed {best_alloc[2]}%")
print(f"  R: {BUDGET * best_alloc[0] / 100:>7,.0f} seashells")
print(f"  S: {BUDGET * best_alloc[1] / 100:>7,.0f} seashells")
print(f"  Sp:{BUDGET * best_alloc[2] / 100:>7,.0f} seashells")
print(f"  ESTIMATED PAYOFF: {best_payoff:,.0f}")
print(f"  vs current (22/75/3): {current_payoff:,.0f}  ({(best_payoff/current_payoff - 1) * 100:+.1f}%)")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# SENSITIVITY: test different Speed multiplier strengths
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SENSITIVITY: if Speed multiplier is STRONGER or WEAKER")
print("=" * 70)
print(f"{'Speed coef':<15}{'Optimal R%':<12}{'Optimal S%':<12}{'Optimal Sp%':<12}{'Payoff':<12}")
print("-" * 63)

for speed_coef in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]:
    def speed_mult_test(Sp):
        if Sp <= 0: return 1.0
        return 1.0 + speed_coef * np.sqrt(Sp)

    def payoff_test(alloc):
        r, s, sp = alloc
        return research_output(BUDGET*r/100) * scale_mult(BUDGET*s/100) * speed_mult_test(BUDGET*sp/100)

    best_p = 0
    best_a = None
    for r in range(1, 99):
        for s in range(1, 100 - r):
            sp = 100 - r - s
            if sp < 1: continue
            p = payoff_test([r, s, sp])
            if p > best_p:
                best_p = p
                best_a = [r, s, sp]
    print(f"{speed_coef:<15.3f}{best_a[0]:<12}{best_a[1]:<12}{best_a[2]:<12}{best_p:>12,.0f}")

print()

# ═══════════════════════════════════════════════════════════════════════════════
# GUIDANCE
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("BHAI'S GUIDANCE")
print("=" * 70)
print("""
Without knowing the exact Speed forecast, the SAFE allocation is:

  KEY PRINCIPLE: Cobb-Douglas / multiplicative payoffs favor BALANCE,
                 not extreme allocations."""
