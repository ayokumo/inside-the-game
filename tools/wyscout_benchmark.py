"""Compute per match benchmark numbers from the Wyscout public dataset.

Download (CC BY 4.0): https://figshare.com/collections/Soccer_match_event_dataset/4415000/2
Unzip events.zip, then:  python tools/wyscout_benchmark.py /path/to/events

Reads every events_*.json file and writes data/reference/wyscout_benchmark.json
with mean and sd per match (both teams combined) for passes, pass accuracy, shots,
goals, fouls and corners.

Tag ids used: 101 goal, 102 own goal, 1801 accurate pass, 1802 not accurate pass.
Verify them against tags2name.csv in the dataset download before trusting the output.
"""
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

GOAL, OWN_GOAL, ACCURATE, NOT_ACCURATE = 101, 102, 1801, 1802


def per_match(events):
    m = defaultdict(lambda: defaultdict(int))
    for e in events:
        c = m[e["matchId"]]
        tags = {t["id"] for t in e.get("tags", [])}
        name, sub = e.get("eventName"), e.get("subEventName")
        if name == "Pass":
            c["passes"] += 1
            c["accurate"] += ACCURATE in tags
        elif name == "Shot":
            c["shots"] += 1
        elif name == "Foul":
            c["fouls"] += 1
        elif name == "Free Kick" and sub == "Corner":
            c["corners"] += 1
        if GOAL in tags and OWN_GOAL not in tags:
            c["goals"] += 1
    return m


def summarise(m):
    rows = []
    for c in m.values():
        if c["passes"] == 0:
            continue
        rows.append({"passes": c["passes"], "pass_accuracy": c["accurate"] / c["passes"], "shots": c["shots"],
                     "goals": c["goals"], "fouls": c["fouls"], "corners": c["corners"]})
    out = {k: {"mean": st.mean(r[k] for r in rows), "sd": st.pstdev(r[k] for r in rows)} for k in rows[0]}
    out["_matches"] = len(rows)
    return out


def main(path):
    events = []
    for f in sorted(Path(path).glob("events_*.json")):
        events += json.loads(f.read_text())
    res = summarise(per_match(events))
    out = Path(__file__).resolve().parents[1] / "data" / "reference" / "wyscout_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
