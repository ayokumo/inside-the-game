"""Match loading. This module only ever reads data/matches, never data/truth."""
from __future__ import annotations

import bisect
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
_CACHE: dict = {}


class Match:
    def __init__(self, events, meta):
        self.events, self.meta = events, meta
        self.ts = [e["t"] for e in events]
        self.by_id = {e["event_id"]: e for e in events}
        self.match_id = meta["match_id"]
        self.home, self.away = meta["home"]["name"], meta["away"]["name"]
        self.players = {}
        for side in ("home", "away"):
            for p in meta[side]["players"]:
                self.players[p["player_id"]] = {"name": p["name"], "team": meta[side]["name"],
                                                "side": side, "group": p["group"]}
        self.duration = meta["duration_s"]

    def other(self, team):
        return self.away if team == self.home else self.home

    def side_name(self, side):
        return self.home if side == "home" else self.away

    def window(self, t0, t1):
        """Events with t0 <= t <= t1."""
        return self.events[bisect.bisect_left(self.ts, t0):bisect.bisect_right(self.ts, t1)]

    def attacks_positive(self, team, period):
        is_home = team == self.home
        return is_home if period == 1 else not is_home


def load_match(match_id, root=None):
    base = Path(root) if root else DATA
    key = (str(base), match_id)
    if key not in _CACHE:
        d = base / "matches" / match_id
        meta = json.loads((d / "meta.json").read_text())
        with open(d / "events.jsonl") as f:
            events = [json.loads(line) for line in f]
        _CACHE[key] = Match(events, meta)
    return _CACHE[key]


def list_matches(root=None):
    base = (Path(root) if root else DATA) / "matches"
    out = []
    for d in sorted(base.iterdir()):
        if (d / "meta.json").exists():
            m = json.loads((d / "meta.json").read_text())
            out.append({"match_id": m["match_id"], "home": m["home"]["name"], "away": m["away"]["name"],
                        "duration_s": m["duration_s"]})
    return out
