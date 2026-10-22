"""Run one real moment through the live agents. Needs: the MCP server running, az login, and two env vars.
Usage:  python3 scripts/live_check.py SYN-0001
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.pipeline import adapt, commentate
from agents.runner import AgentRunner
from intel.data import load_match
from intel.moments import detect_moments


async def main(match_id):
    m = load_match(match_id)
    moments = detect_moments(m, m.duration)
    kinds = {}
    for x in moments:                                   # up to two of each kind, so every path is exercised
        if sum(1 for y in kinds.values() if y["kind"] == x["kind"]) < 2:
            kinds[x["id"]] = x
    stats = {"moments": 0, "first_try": 0, "rewritten": 0, "fallback": 0}
    async with AgentRunner() as run:
        print("backend:", run.backend)
        for mom in kinds.values():
            c = (await commentate(m, mom["t"] + 1, run, since_t=mom["t"] - 1))
            c = next((x for x in c if x.moment_id == mom["id"]), None)
            if c is None:
                continue
            stats["moments"] += 1
            stats["fallback" if c.used_fallback else ("first_try" if c.attempts == 1 else "rewritten")] += 1
            print(f"\n[{mom['kind']}] {c.moment_id} attempts={c.attempts} fallback={c.used_fallback}")
            print("  TEXT:", c.text)
            for k in c.catches:
                print("  CAUGHT (attempt %d):" % k["attempt"], json.dumps(k["problems"], default=str)[:300])
            if mom["kind"] == "goal":
                for persona, loc in (("casual", "en"), ("accessibility", "en"), ("analyst", "de")):
                    r = await adapt(m, c, run, persona, loc)
                    print(f"  {persona}/{loc}:", r["text"], r["notes"])
    print("\nSUMMARY", stats)


asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "SYN-0001"))
