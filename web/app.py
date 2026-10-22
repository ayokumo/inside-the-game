"""Web app: replay API, commentary service and the single page UI."""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from agents.pipeline import PERSONAS
from intel import claims as C
from intel.data import list_matches, load_match
from intel.stats import control_at, player_stats, team_stats
from web.service import Service

STATIC = Path(__file__).parent / "static"
svc = Service()
STATE = {"mode": "cached or template only", "backend": None}


@asynccontextmanager
async def lifespan(app):
    """Use live agents when configured (AGENTS=live plus model settings), otherwise serve cache and templates."""
    runner = None
    if os.environ.get("AGENTS") == "live":
        try:
            from agents.runner import AgentRunner
            runner = AgentRunner()
            await runner.__aenter__()
            svc.runner, STATE["mode"], STATE["backend"] = runner, "live agents", runner.backend
        except Exception as ex:                      # keep serving cache if the model or MCP server is unavailable
            print("live agents unavailable:", repr(ex))
    yield
    if runner:
        await runner.__aexit__(None, None, None)


def _match(request):
    try:
        return load_match(request.path_params["mid"])
    except (OSError, ValueError):
        raise HTTPException(404, "unknown match")


def _t(request, m, name="as_of", default=None):
    try:
        v = float(request.query_params.get(name, default))
    except (TypeError, ValueError):
        raise HTTPException(400, f"{name} must be a number")
    return max(0.0, min(v, m.duration))


async def config(request):
    return JSONResponse({"mode": STATE["mode"], "backend": STATE["backend"], "personas": list(PERSONAS),
                         "locales": list(C.LOCALES)})


async def matches(request):
    out = []
    for x in list_matches():
        m = load_match(x["match_id"])
        out.append(dict(x, final_score=m.meta["final_score"], home_players=m.meta["home"]["players"],
                        away_players=m.meta["away"]["players"]))
    return JSONResponse(out)


async def pitch_events(request):
    """Compact events for drawing the pitch. The browser only draws up to the playhead."""
    m = _match(request)
    keep = ("pass", "shot", "pressure", "tackle", "interception", "card", "substitution", "corner", "foul")
    return JSONResponse({"duration": m.duration, "home": m.home, "away": m.away, "events": [
        [e["event_id"], round(e["t"], 1), e["type"], 0 if e["team"] == m.home else 1, e.get("start"), e.get("end"),
         e.get("outcome"), e.get("xg"), e.get("player"), e["period"]] for e in m.events if e["type"] in keep]})


async def state(request):
    m = _match(request)
    t = _t(request, m)
    stats = team_stats(m, t)
    evs = m.window(0.0, t)
    score = next((e["score"] for e in reversed(evs) if e.get("score")), [0, 0])
    c = control_at(m, t)
    keys = ("shots", "shots_on_target", "xg", "possession_pct", "pass_accuracy_pct", "field_tilt_pct", "corners", "fouls")
    return JSONResponse({"as_of": t, "score": score, "clock": evs[-1]["clock"] if evs else "1",
                         "control": {"label": c["label"], "team": c["team"]},
                         "stats": {k: {key: stats[tm][key] for key in keys} for k, tm in (("home", m.home), ("away", m.away))},
                         "moments": [{k: x[k] for k in ("id", "t", "kind", "team", "importance")} for x in svc.moments(m, t)],
                         "finished": t >= m.duration})


async def player(request):
    m = _match(request)
    t = _t(request, m)
    pid = request.query_params.get("player_id", "")
    if pid not in m.players:
        raise HTTPException(404, "unknown player")
    return JSONResponse(dict(player_stats(m, t).get(pid, {}), name=m.players[pid]["name"], team=m.players[pid]["team"]))


async def moment_view(request):
    m = _match(request)
    t = _t(request, m)
    moment = next((x for x in svc.moments(m, t) if x["id"] == request.path_params["moment_id"]), None)
    if moment is None:
        raise HTTPException(404, "that moment has not happened yet at as_of")
    q = request.query_params
    persona = q.get("persona", "casual")
    if persona not in PERSONAS:
        raise HTTPException(400, "unknown persona")
    locale = q.get("locale", "en")
    if locale not in C.LOCALES:
        raise HTTPException(400, "unsupported locale")
    fav = q.get("favorite_team") or None
    if fav and fav not in (m.home, m.away):
        raise HTTPException(400, "favorite_team must be one of the two clubs")
    pid = q.get("player_id") or None
    if pid and pid not in m.players:
        raise HTTPException(400, "unknown player")
    return JSONResponse(await svc.view(m, moment, persona, locale, fav, pid))


async def index(request):
    return FileResponse(STATIC / "index.html")


app = Starlette(lifespan=lifespan, routes=[
    Route("/", index), Route("/api/config", config), Route("/api/matches", matches),
    Route("/api/match/{mid}/events", pitch_events), Route("/api/match/{mid}/state", state),
    Route("/api/match/{mid}/player", player), Route("/api/match/{mid}/moment/{moment_id:path}", moment_view),
    Mount("/static", StaticFiles(directory=STATIC), name="static")])
