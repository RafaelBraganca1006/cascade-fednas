"""Offline: how much of the TOP shell (D3, XL-exclusive) actually arrives?

Same trick as analysis/pick_deadline.py -- the bandwidth chains are seeded and
deterministic and the shell payloads are constants, so this is answerable with
no GPU. Compares the two live tx_modes on the one quantity the Pareto spread
depends on.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/Documents/fednas/dynfednas_v2"))
from fedml_api.standalone.superfednas.comm_sim.bandwidth import (
    MarkovBandwidth, assign_connectivity_tiers,
)

MB = 8e6
TIERS = [41.6, 98.6, 137.6, 286.9]           # cumulative payload, MB
SHELL = [TIERS[0]] + [TIERS[i] - TIERS[i - 1] for i in range(1, 4)]
T, N, PER_ROUND, ROUNDS = 40, 20, 8, 200

tiers = assign_connectivity_tiers({c: np.ones(10) for c in range(N)}, corr=0.0, seed=0)
bw = MarkovBandwidth.tiered_continuous(N, ROUNDS, tiers, seed=0, jitter_sigma=0.45)

budgets = []
for r in range(ROUNDS):
    for c in range(N):
        budgets.append(bw.bandwidth(c, r) * T / MB)     # MB deliverable

# A client only streams shells 0..train_tier. Blind TS_all_random draws an arch
# per client per round; approximate the XL-assigned case (train_tier=3), which
# is the only case where D3 units are even in the stream.
nested_s3 = sum(1 for b in budgets if b >= TIERS[3]) / len(budgets)
# Block stream: D3 is sent LAST, so its coverage is the leftover budget after
# the whole L payload, capped at 1.
blocks_s3 = sum(min(max(b - TIERS[2], 0.0) / SHELL[3], 1.0) for b in budgets) / len(budgets)
part_only = sum(1 for b in budgets if TIERS[2] < b < TIERS[3]) / len(budgets)

print(f"shell payloads MB: D0={SHELL[0]:.1f} D1={SHELL[1]:.1f} "
      f"D2={SHELL[2]:.1f} D3={SHELL[3]:.1f}")
print(f"budget MB: mean={sum(budgets)/len(budgets):.0f} "
      f"p50={sorted(budgets)[len(budgets)//2]:.0f} max={max(budgets):.0f}\n")
print(f"D3 coverage, nested (all-or-nothing per shell): {nested_s3:6.1%}")
print(f"D3 coverage, blocks  (partial shell allowed)  : {blocks_s3:6.1%}")
print(f"  client-rounds that reach L but not XL       : {part_only:6.1%}"
      "  <- delivers 0 of D3 under nested, partial under blocks")
