"""Commentary service for the web app. Agents run once per moment and persona view, and results are cached on disk
(data/cache) so the public demo is instant and does not depend on a live model. Without a runner and without a cache
entry the app falls back to the deterministic template and says so on screen."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agents import pipeline as P
from intel import claims as C
from intel import store
from intel.data import Match, load_match
from intel.moments import detect_moments

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
SHOW_KINDS = {"goal", "red_card", "big_chance_missed", "press_spike", "control_shift"}


def _slug(x):
    return "".join(ch if ch.isalnum() else "_" for ch in str(x or "none"))


class Service:
    def __init__(self, runner=None, cache_dir: Path | None = None):
        self.runner, self.cache = runner, Path(cache_dir or CACHE)
        self._moments, self._locks = {}, {}

    # ---- moments (computed once, filtered by time; prefix stability is covered by tests)
    def moments(self, match: Match, as_of: float) -> list:
        if match.match_id not in self._moments:
            self._moments[match.match_id] = [m for m in detect_moments(match, match.duration) if m["kind"] in SHOW_KINDS]
        return [m for m in self._moments[match.match_id] if m["t"] <= as_of]

    def _lock(self, key):
        return self._locks.setdefault(key, asyncio.Lock())

    def _read(self, path: Path):
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None

    def _write(self, path: Path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, default=str))

    # ---- base commentary: narrator + checker loop for one moment
    async def base(self, match: Match, moment: dict) -> dict:
        key = f"{match.match_id}__{moment['id'].split(':')[-1]}"
        path = self.cache / "base" / f"{key}.json"
        async with self._lock(key):
            hit = self._read(path)
            if hit:
                return dict(hit, source="cached agents")
            if self.runner:
                c = await P.narrate_moment(match, moment, moment["t"], self.runner)
                data = {"skeleton": c.skeleton, "claims": c.claims, "verified": c.verified, "catches": c.catches,
                        "attempts": c.attempts, "used_fallback": c.used_fallback}
                self._write(path, data)
                store.state_put("case_files", key, dict(data, moment=moment["id"], as_of=moment["t"]))   # shared case file
                return dict(data, source="agents")
            cl, sk = P.template_for(moment, match)
            res = C.verify_claims(match, cl, moment["t"])
            return {"skeleton": sk, "claims": cl, "verified": {r["id"]: r["verified"] for r in res if r["ok"]},
                    "catches": [], "attempts": 0, "used_fallback": True, "source": "template (offline)"}

    # ---- one persona and language view of a moment
    async def view(self, match: Match, moment: dict, persona="casual", locale="en", favorite_team=None, player_id=None) -> dict:
        base = await self.base(match, moment)
        sk, notes, source = base["skeleton"], [], base["source"]
        vkey = "__".join([match.match_id, moment["id"].split(":")[-1], persona, locale, _slug(favorite_team), _slug(player_id)])
        vpath = self.cache / "views" / f"{vkey}.json"
        async with self._lock(vkey):
            hit = self._read(vpath)
            if not hit and not self.runner:      # personalised variant not cached: use the closest cached one
                base_key = [match.match_id, moment["id"].split(":")[-1], persona, locale]
                for fav_s, pl_s in ((_slug(None), _slug(player_id)), (_slug(None), _slug(None))):
                    hit = self._read(self.cache / "views" / ("__".join(base_key + [fav_s, pl_s]) + ".json"))
                    if hit:
                        break
            if hit:
                sk, notes, source = hit["skeleton"], hit["notes"], "cached agents"
            elif self.runner and sk:
                c = P.Commentary(moment["id"], sk, base["claims"], base["verified"], "", base["catches"], base["attempts"],
                                 base["used_fallback"], moment)
                out = await P.adapt(match, c, self.runner, persona, locale, favorite_team, player_id)
                sk, notes, source = out["skeleton"], out["notes"], "agents"
                self._write(vpath, {"skeleton": sk, "notes": notes})
            elif locale != "en":
                notes.append("translation needs the agents; showing English")
                locale = "en"
        loc = locale if locale in C.LOCALES else "en"
        parts = C.render_parts(match, sk, base["verified"], loc) if sk else []
        used = {p["claim"] for p in parts if p["claim"]}
        claimed = {c["id"]: c for c in base["claims"]}
        receipts = {}
        for cid in used:
            v = base["verified"][cid]
            r = {"verified": v, "claimed_value": claimed.get(cid, {}).get("value")}
            if v["kind"] == "stat":
                r["events_in_window"] = len(match.window(v["from_t"], v["to_t"]))
            if v["kind"] == "event":
                r["event"] = match.by_id.get(v["event_id"])
            receipts[cid] = r
        return {"moment": {k: moment[k] for k in ("id", "t", "clock", "kind", "team", "importance", "player", "player_id")},
                "parts": parts, "text": "".join(p["text"] for p in parts), "receipts": receipts,
                "catches": base["catches"], "attempts": base["attempts"], "used_fallback": base["used_fallback"],
                "persona": persona, "locale": loc, "notes": notes, "source": source}
