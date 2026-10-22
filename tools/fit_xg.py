"""Fit the logistic xG curve used by the simulator.

The anchors below are approximate open play, foot, unpressured values for typical
shot positions. They are an engineering approximation, NOT fitted to a real dataset.
The realism check (tools/realism.py, tools/wyscout_benchmark.py) is what compares
the simulator's aggregate output to real match data.

Run: python tools/fit_xg.py   (prints coefficients to paste into matchsim/xg.py)
"""
import math
import numpy as np

POST_HALF = 3.66


def geometry(dx, dy):
    dx = max(dx, 0.5)
    dist = math.hypot(dx, dy)
    a1 = math.atan2(POST_HALF - dy, dx)
    a2 = math.atan2(-POST_HALF - dy, dx)
    return dist, abs(a1 - a2)


# (distance along pitch to goal line, lateral offset from goal centre, xG)
ANCHORS = [
    (6, 0, 0.36),
    (11, 0, 0.15),
    (16.5, 0, 0.07),
    (18, 8, 0.04),
    (25, 0, 0.03),
    (30, 0, 0.02),
    (4, 10, 0.05),
    (10, 10, 0.05),
    (12, 5, 0.10),
    (8, 3, 0.20),
    (20, 4, 0.045),
]

rows, ys = [], []
for dx, dy, p in ANCHORS:
    d, a = geometry(dx, dy)
    rows.append([1.0, d, a])
    ys.append(math.log(p / (1 - p)))

coef, *_ = np.linalg.lstsq(np.array(rows), np.array(ys), rcond=None)
print("B0=%.4f B1=%.4f B2=%.4f" % tuple(coef))
for (dx, dy, p), r in zip(ANCHORS, rows):
    z = float(np.dot(coef, r))
    print(f"dx={dx:5.1f} dy={dy:4.1f} target={p:.3f} fitted={1/(1+math.exp(-z)):.3f}")
