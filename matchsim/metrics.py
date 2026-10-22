"""Aggregate match metrics used by the realism check and the tests."""
from __future__ import annotations


def match_metrics(events, meta):
    """Per match totals (both teams combined) plus per team splits."""
    out = {}
    for side in ("home", "away"):
        name = meta[side]["name"]
        passes = [e for e in events if e["type"] == "pass" and e["team"] == name]
        shots = [e for e in events if e["type"] == "shot" and e["team"] == name]
        out[side] = {
            "passes": len(passes),
            "completion": sum(e["outcome"] == "complete" for e in passes) / max(1, len(passes)),
            "shots": len(shots),
            "on_target": sum(e["outcome"] in ("goal", "saved") for e in shots),
            "goals": sum(e["outcome"] == "goal" for e in shots),
            "xg": sum(e["xg"] for e in shots),
            "tackles": sum(1 for e in events if e["type"] == "tackle" and e["team"] == name),
            "fouls": sum(1 for e in events if e["type"] == "foul" and e["team"] == name),
            "corners": sum(1 for e in events if e["type"] == "corner" and e["team"] == name),
            "yellows": sum(1 for e in events if e["type"] == "card" and e["team"] == name and e["card_type"] == "yellow"),
            "reds": sum(1 for e in events if e["type"] == "card" and e["team"] == name and e["card_type"] != "yellow"),
        }
    h, a = out["home"], out["away"]
    tot_passes = h["passes"] + a["passes"]
    out["total"] = {
        "passes": tot_passes,
        "completion": (h["completion"] * h["passes"] + a["completion"] * a["passes"]) / max(1, tot_passes),
        "shots": h["shots"] + a["shots"],
        "goals": h["goals"] + a["goals"],
        "xg": h["xg"] + a["xg"],
        "possession_home": h["passes"] / max(1, tot_passes),
        "events": len(events),
    }
    return out
