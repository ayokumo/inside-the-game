"""Sweep shot propensity constants and report aggregate output. Used once to tune
matchsim.engine.SHOT_BASE / SHOT_DECAY / GOAL_SCALE. Run: python tools/tune_shots.py"""
import statistics as st
import sys
sys.path.insert(0, ".")
from matchsim import engine
from matchsim.metrics import match_metrics

def run(base, decay, scale, n=60):
    engine.SHOT_BASE, engine.SHOT_DECAY, engine.GOAL_SCALE = base, decay, scale
    ms = []
    for seed in range(7000, 7000 + n):
        r = engine.simulate("NBR", "HLW", seed, "SYN-T")
        ms.append(match_metrics(r["events"], r["meta"])["total"])
    return (st.mean(m["shots"] for m in ms), st.mean(m["xg"] for m in ms), st.mean(m["goals"] for m in ms))

if __name__ == "__main__":
    for base, decay in [(0.30, 13), (0.36, 11), (0.40, 10), (0.44, 9.5), (0.50, 9)]:
        sh, xg, g = run(base, decay, 0.88)
        print(f"base={base:.2f} decay={decay:>4} shots={sh:5.1f} xg={xg:4.2f} xg/shot={xg/sh:.3f} goals={g:4.2f}")
