"""Fill the commentary cache with real agent output so the public demo is instant and never depends on a live model.
Resumable: anything already cached is skipped. Needs the MCP server running plus Foundry settings (see live_check.py).
Usage: python3 scripts/pregenerate.py [MATCH_ID ...]
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.runner import AgentRunner
from intel.data import list_matches, load_match
from web.service import Service

JOBS = [("analyst", "en", None), ("casual", "en", None), ("player_focus", "en", None), ("accessibility", "en", None)] + \
       [("casual", loc, None) for loc in ("de", "fr", "es", "pt")] + [("accessibility", "de", None)]
CONCURRENCY = 3


async def main(ids):
    t0, done, failed = time.time(), 0, 0
    async with AgentRunner() as run:
        svc = Service(run)
        print("backend:", run.backend)
        sem = asyncio.Semaphore(CONCURRENCY)

        async def one(m, mom, persona, loc, fav):
            nonlocal done, failed
            async with sem:
                try:
                    await svc.view(m, mom, persona, loc, fav, None)
                    done += 1
                except Exception as ex:
                    failed += 1
                    print("FAILED", mom["id"], persona, loc, fav, repr(ex)[:160])
                if (done + failed) % 10 == 0:
                    print(f"  {done} done, {failed} failed, {int(time.time() - t0)}s", flush=True)

        for mid in ids:
            m = load_match(mid)
            moments = svc.moments(m, m.duration)
            print(mid, len(moments), "moments")
            for mom in moments:                       # base commentary first, so views can reuse it
                await svc.base(m, mom)
            jobs = [(mom, p, l, None) for mom in moments for p, l, _ in JOBS]
            jobs += [(mom, "casual", "en", team) for mom in moments for team in (m.home, m.away)]
            await asyncio.gather(*(one(m, *j) for j in jobs))
    print(f"finished: {done} views, {failed} failed, {int(time.time() - t0)}s")


asyncio.run(main(sys.argv[1:] or [x["match_id"] for x in list_matches()]))
