"""Command line: generate matches, write them to disk, scan seeds for narrative.

python -m matchsim.cli generate --home NBR --away HLW --seed 7 --id SYN-0001 --out data
python -m matchsim.cli select --out data          (scan seeds, save the 3 demo matches)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engine import simulate
from .metrics import match_metrics


def write_match(result, root):
    mid = result["meta"]["match_id"]
    d = Path(root) / "matches" / mid
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "events.jsonl", "w") as f:
        for e in result["events"]:
            f.write(json.dumps(e, separators=(",", ":")) + "\n")
    (d / "meta.json").write_text(json.dumps(result["meta"], indent=2))
    # Hidden ground truth lives outside matches/ so agent tools never read it.
    tdir = Path(root) / "truth"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / f"{mid}.json").write_text(json.dumps(result["truth"], indent=2))
    return d


def narrative_score(result, want_red=False):
    """Score how watchable a match is. Used to pick demo matches from many seeds.
    The matches themselves are emergent, this only chooses which seeds to keep."""
    ev, truth = result["events"], result["truth"]
    goals = [e for e in ev if e["type"] == "shot" and e["outcome"] == "goal"]
    total = len(goals)
    score, notes = 0.0, []
    if 2 <= total <= 6:
        score += 1
    if total == 0:
        score -= 3
    h = a = 0
    leader, changes = 0, 0
    for g in goals:
        if g["team"] == result["meta"]["home"]["name"]:
            h += 1
        else:
            a += 1
        lead = (h > a) - (h < a)
        if lead != 0 and leader != 0 and lead != leader:
            changes += 1
        if lead != 0:
            leader = lead
    if changes:
        score += 3 * min(changes, 2)
        notes.append(f"{changes} lead change(s)")
    if truth["red_cards"]:
        score += 2 if want_red else -2  # keep the three demo matches different from each other
        notes.append("red card")
    if any(g["minute"] >= 80 for g in goals):
        score += 1
        notes.append("late goal")
    if abs(h - a) >= 4:
        score -= 2
    best = 0.0
    for w in truth["tactical_windows"]:
        m = w["measured"]
        if w["mode"] == "high_press" and m["press_intensity_out"] and m["press_intensity_in"]:
            best = max(best, m["press_intensity_in"] / m["press_intensity_out"] - 1)
    if best > 0.3:
        score += 2
        notes.append(f"clear press window (+{best:.0%} press intensity)")
    return score, notes


def select(root, pairs, n_seeds=150, start_seed=1000, want_red=(False, False, True)):
    log = {"method": "scan seeds, keep the highest narrative score per fixture. Matches are emergent, not scripted.",
           "fixtures": []}
    for idx, (home, away) in enumerate(pairs, start=1):
        scored = []
        for seed in range(start_seed + idx * 1000, start_seed + idx * 1000 + n_seeds):
            r = simulate(home, away, seed, f"SYN-{idx:04d}")
            s, notes = narrative_score(r, want_red[idx - 1])
            scored.append((s, seed, notes, r["meta"]["final_score"]))
        scored.sort(key=lambda t: -t[0])
        s, seed, notes, fs = scored[0]
        result = simulate(home, away, seed, f"SYN-{idx:04d}")
        write_match(result, root)
        log["fixtures"].append({
            "match_id": f"SYN-{idx:04d}", "home": home, "away": away, "chosen_seed": seed,
            "narrative_score": s, "notes": notes, "final_score": fs, "seeds_scanned": n_seeds,
            "runner_up_scores": [t[0] for t in scored[1:6]],
            "median_score": sorted(t[0] for t in scored)[len(scored) // 2]})
        print(f"SYN-{idx:04d} {home}-{away} seed {seed} score {s} {notes} final {fs}")
    (Path(root) / "selection_log.json").write_text(json.dumps(log, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--home", required=True)
    g.add_argument("--away", required=True)
    g.add_argument("--seed", type=int, required=True)
    g.add_argument("--id", default="SYN-0001")
    g.add_argument("--out", default="data")
    s = sub.add_parser("select")
    s.add_argument("--out", default="data")
    s.add_argument("--seeds", type=int, default=150)
    args = ap.parse_args()
    if args.cmd == "generate":
        r = simulate(args.home, args.away, args.seed, args.id)
        d = write_match(r, args.out)
        print(d, r["meta"]["final_score"], match_metrics(r["events"], r["meta"])["total"])
    else:
        select(args.out, [("NBR", "HLW"), ("CAS", "PAV"), ("PAV", "NBR")], args.seeds)


if __name__ == "__main__":
    main()
