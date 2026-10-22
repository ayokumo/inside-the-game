"""MCP server: the only door agents use to reach match facts. Every tool takes `as_of` (seconds of playing
time) and cannot see past it. It reads data/matches only, never data/truth."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server.mcpserver import MCPServer

from intel import claims as C
from intel import store
from intel.data import list_matches as _list, load_match
from intel.moments import detect_moments, top_moments
from intel.stats import CFG, control_at, player_stats, team_stats, threshold_crossings, window_features

server = MCPServer("match-intel")


def _m(match_id, as_of):
    m = load_match(match_id)
    if as_of < 0:
        raise ValueError("as_of must be >= 0")
    return m, min(float(as_of), m.duration)


@server.tool()
def list_matches() -> list:
    """List available synthetic matches."""
    return _list()


@server.tool()
def get_match_state(match_id: str, as_of: float) -> dict:
    """Score, clock, and control label at as_of."""
    m, t = _m(match_id, as_of)
    evs = m.window(0.0, t)
    last = evs[-1] if evs else None
    score = next((e["score"] for e in reversed(evs) if e.get("score")), [0, 0])
    c = control_at(m, t)
    return {"match_id": match_id, "as_of": t, "home": m.home, "away": m.away, "score": score,
            "clock": last["clock"] if last else "0:00", "control": {"label": c["label"], "team": c["team"]},
            "finished": t >= m.duration}


@server.tool()
def get_team_stats(match_id: str, as_of: float, from_t: float = 0.0) -> dict:
    """Team statistics for [from_t, as_of]."""
    m, t = _m(match_id, as_of)
    return team_stats(m, t, from_t)


@server.tool()
def get_player_stats(match_id: str, as_of: float, player_id: str = "") -> dict:
    """Player stats from kickoff to as_of. Leave player_id empty for everyone."""
    m, t = _m(match_id, as_of)
    ps = player_stats(m, t)
    return ps.get(player_id, {}) if player_id else ps


@server.tool()
def get_moments(match_id: str, as_of: float, top_k: int = 0) -> list:
    """Detected key moments up to as_of. top_k > 0 returns only the most important ones."""
    m, t = _m(match_id, as_of)
    return top_moments(m, t, top_k) if top_k > 0 else detect_moments(m, t)


@server.tool()
def get_events(match_id: str, as_of: float, from_t: float = 0.0, types: list[str] | None = None, limit: int = 200) -> list:
    """Raw events in [from_t, as_of], optionally filtered by type. Capped at `limit`, most recent kept."""
    m, t = _m(match_id, as_of)
    evs = [e for e in m.window(from_t, t) if not types or e["type"] in types]
    return evs[-min(limit, 500):]


@server.tool()
def get_window_features(match_id: str, as_of: float, from_t: float) -> dict:
    """Rolling window picture (possession, tilt, turnovers) for [from_t, as_of]."""
    m, t = _m(match_id, as_of)
    return window_features(m, from_t, t)


@server.tool()
def get_thresholds_crossed(match_id: str, as_of: float) -> list:
    """Distance and top speed thresholds players have crossed by as_of."""
    m, t = _m(match_id, as_of)
    return threshold_crossings(m, t)


@server.tool()
def get_thresholds() -> dict:
    """The configured thresholds and detector settings, so numbers can be explained."""
    return CFG


@server.tool()
def verify_claims(match_id: str, as_of: float, claims: list[dict]) -> list:
    """Verify structured claims by recomputing them. Returns ok, reason, and verified facts for each."""
    m, t = _m(match_id, as_of)
    return C.verify_claims(m, claims, t)


@server.tool()
def render_text(match_id: str, as_of: float, skeleton: str, claims: list[dict], locale: str = "en") -> dict:
    """Verify claims then render the skeleton from verified values only. Never raises on bad input; reports problems."""
    m, t = _m(match_id, as_of)
    if locale not in C.LOCALES:
        return {"ok": False, "problems": [f"unsupported locale {locale}"], "text": None}
    res = C.verify_claims(m, claims, t)
    verified = {r["id"]: r["verified"] for r in res if r["ok"]}
    problems = C.check_skeleton(skeleton, verified)
    failed = [{"id": r["id"], "reason": r["reason"]} for r in res if not r["ok"]]
    if problems or failed:
        return {"ok": False, "problems": problems, "failed_claims": failed, "text": None}
    try:
        return {"ok": True, "problems": [], "failed_claims": [], "text": C.render(m, skeleton, verified, locale)}
    except (ValueError, KeyError) as ex:
        return {"ok": False, "problems": [f"render failed: {ex}"], "failed_claims": [], "text": None}


@server.tool()
def state_get(namespace: str, key: str) -> dict:
    """Read shared agent state."""
    return store.state_get(namespace, key)


@server.tool()
def state_put(namespace: str, key: str, value: dict, expected_version: int | None = None) -> dict:
    """Write shared agent state with optional compare and swap."""
    return store.state_put(namespace, key, value, expected_version)


def app():
    return server.streamable_http_app()


if __name__ == "__main__":
    server.run("streamable-http")
