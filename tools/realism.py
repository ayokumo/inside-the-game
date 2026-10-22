"""Realism check: 20 UNSELECTED synthetic matches vs the Wyscout benchmark.

Matches are simulated from fixed seeds with no narrative filtering, so this shows
the simulator's baseline behaviour, not the hand picked demo matches.

1. python tools/wyscout_benchmark.py /path/to/wyscout/events  -> data/reference/wyscout_benchmark.json
2. python tools/realism.py                                      -> data/realism_report.md
Without step 1 the report states that the real reference is pending.
"""
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matchsim import simulate  # noqa: E402
from matchsim.metrics import match_metrics  # noqa: E402

PAIRS = [("NBR", "HLW"), ("CAS", "PAV"), ("PAV", "NBR"), ("HLW", "CAS")]
ROOT = Path(__file__).resolve().parents[1] / "data"


def synthetic(n=20, seed0=9000):
    rows = []
    for i in range(n):
        h, a = PAIRS[i % len(PAIRS)]
        r = simulate(h, a, seed0 + i, f"SYN-R{i:02d}")
        m = match_metrics(r["events"], r["meta"])
        both = lambda k: m["home"][k] + m["away"][k]
        rows.append({"passes": m["total"]["passes"], "pass_accuracy": m["total"]["completion"],
                     "shots": m["total"]["shots"], "goals": m["total"]["goals"],
                     "fouls": both("fouls"), "corners": both("corners")})
    return rows


def summarise(rows):
    return {k: {"mean": st.mean(r[k] for r in rows), "sd": st.pstdev(r[k] for r in rows)} for k in rows[0]}


def main():
    syn = summarise(synthetic())
    ref_path = ROOT / "reference" / "wyscout_benchmark.json"
    ref = json.loads(ref_path.read_text()) if ref_path.exists() else None
    lines = ["# Realism check", "",
             "20 unselected synthetic matches (both teams combined) against the Wyscout public dataset.", ""]
    if ref is None:
        lines += ["**Real reference not generated yet.** Run `tools/wyscout_benchmark.py` on the Wyscout download "
                  "and rerun this script. Until then the numbers below are the synthetic side only.", ""]
    lines += ["| metric | synthetic mean (sd) | Wyscout mean (sd) | gap in Wyscout sds |", "| --- | --- | --- | --- |"]
    for k, v in syn.items():
        f = (lambda x: f"{x:.3f}") if k == "pass_accuracy" else (lambda x: f"{x:.1f}")
        if ref and k in ref:
            gap = (v["mean"] - ref[k]["mean"]) / ref[k]["sd"] if ref[k]["sd"] else float("nan")
            lines.append(f"| {k} | {f(v['mean'])} ({f(v['sd'])}) | {f(ref[k]['mean'])} ({f(ref[k]['sd'])}) | {gap:+.2f} |")
        else:
            lines.append(f"| {k} | {f(v['mean'])} ({f(v['sd'])}) | pending | pending |")
    lines += ["", "Definitions differ slightly: Wyscout counts free kick shots and penalties under other event names, "
              "and has no event for a pass that was blocked. Treat gaps under about 1 sd as consistent."]
    out = ROOT / "realism_report.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
