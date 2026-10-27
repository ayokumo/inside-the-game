# Match Intel

*Working title. A multi agent football commentary system where every number has a receipt.*

Built for the Microsoft Premier League Hackathon (Multi Agent System category). All match data is **synthetic**.

## The problem

AI commentary sounds confident whether it is right or wrong. A fan, a coach, or a broadcaster cannot tell a real
statistic from an invented one, and a wrong number in live sport is a trust problem, not a typo.

## The idea: claims with receipts

Agents never write numbers. They write **structured claims** ("Northbridge possession from minute 0 to 53") and a
sentence with placeholders. Code then **recomputes every claim from the raw event log**, rejects anything that does not
match, and builds the final sentence itself. In the app, every number is a button: tap it to see what the writer
claimed, what code recomputed, and which events it came from. A rejected draft is shown to the user as
**"Caught and fixed"**, with the reason.

Because persona and translation agents only ever see the placeholder sentence, they can change tone and language but
cannot change a fact. Numbers are formatted by code per language (for example `0,88` in German).

## Architecture

```
 synthetic match events ──► intel/ (stats, moments, claims verifier)   deterministic, no model
                                   ▲
                                   │ MCP (streamable HTTP, 13 tools, every tool bounded by `as_of`)
        ┌──────────────────────────┴───────────────────────────┐
        │ Microsoft Agent Framework agents on a Foundry model   │
        │  Analyst   picks which new moments matter and why     │
        │  Narrator  writes claims + placeholder sentence       │
        │  Checker   code verifies claims, rules, causes;       │
        │            LLM reviews wording for unsupported facts  │
        │  Persona   rewrites tone (4 views)                    │
        │  Translator  5 languages, placeholders must match     │
        └──────────────────────────┬───────────────────────────┘
                                   ▼
                  web app (Starlette): pitch replay, receipts, caught and fixed
```

**Agents and roles.** Each role is its own Microsoft Agent Framework `Agent` with separate instructions. The Analyst
and Narrator reach match facts only through the MCP server (the same one any client could use). The Checker is two
layers: deterministic code, then an LLM reviewer.

**Coordination.** Our own deterministic pipeline (`agents/pipeline.py`) passes typed messages between agents: Analyst
note, claim menu, draft, feedback. A shared case file per moment is written to the state store exposed over MCP
(`state_get`, `state_put` with version checks). We did not use the framework's built in handoff or workflow builders;
the loop has to be deterministic, so we wrote it directly.

**Failure recovery.** A rejected draft goes back to the Narrator with the exact reasons, up to 2 rewrites. After that a
deterministic template sentence is used and labelled. A bad persona or translation output is dropped and the previous
wording kept.

**No lookahead.** Every query takes `as_of` (seconds of playing time) and cannot see later events. Tests check that
truncating the log gives identical results and that detected moments are a stable prefix as time advances.

## What code guarantees, and what it does not

Guaranteed by code (tested):
- Every number and every named player, team, clock and score shown comes from recomputation or a lookup in the log.
- No digits, percent signs or English number words in model written prose.
- No causal link ("after the red card") unless a detector found that driver for that moment.
- Future windows, unknown events, wrong players and malformed claims are rejected.

**Not guaranteed (stated plainly):**
- The LLM reviewer is probabilistic. It can miss an unsupported phrase. The deterministic word filter helps but is
  English only; translations rely on the reviewer to catch added facts.
- Number words in other languages are not filtered by code, only reviewed.
- The "pressure rise" detector is imperfect: on 40 simulated matches it found 63% of real high press periods, about 44%
  of its alerts were real, 1.2 false alerts per match. It is labelled a lead, not a fact.
- Commentary quality as football writing has not been rated by independent human readers.

Measured results from the real agent runs are in [`docs/eval_results.md`](docs/eval_results.md).

## Synthetic data

Rule 4.7 requires synthetic data. `matchsim/` is a possession chain simulator with hidden causes (tactical modes,
fatigue, red cards, substitutions, match tempo) written to a separate hidden truth log. Shot quality uses a logistic
expected goals model on distance and angle. The agents never read the truth log; it is not in the deployed image and a
test checks the `intel/` and `mcp_server/` code cannot reference it. See [`docs/synthetic_data.md`](docs/synthetic_data.md)
for limits: no penalties or offsides, modelled speeds, curated demo seeds, and a realism comparison script
(`tools/wyscout_benchmark.py`) against public Wyscout data that has **not** been completed for this submission.

## Microsoft and Azure technologies

- Microsoft Agent Framework (Python) for every agent
- Microsoft Foundry model deployment (gpt-4.1-mini) via `FoundryChatClient`
- Model Context Protocol: our server (MCP Python SDK) and the framework's MCP tool client
- Azure Container Apps hosts the public demo
- GitHub Actions builds the container image and GitHub Container Registry stores it; Azure runs it

## Run it

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m pytest -q                                   # 55+ tests
python3 -m uvicorn web.app:app --port 8080              # app, serves cached agent output
```

Live agents: start `python3 mcp_server/server.py`, set `FOUNDRY_PROJECT_ENDPOINT` and `FOUNDRY_MODEL`, run `az login`,
then `AGENTS=live python3 -m uvicorn web.app:app --port 8080`. To fill the cache: `python3 scripts/pregenerate.py`.
Without a model the app shows cached commentary, or a plain template labelled "agents offline".

## Repository map

`matchsim/` simulator, `intel/` verification core, `mcp_server/` MCP tools, `agents/` pipeline and prompts,
`web/` app, `eval/` reports, `tests/`, `data/` synthetic matches and cached agent output.

## Limits of the demo

The pitch shows ball events (passes, shots, pressures), not all 22 players, because the data is event data. Interface
labels are English; commentary is translated. The public demo serves pre generated agent output so it is instant and
does not depend on a live model.

## Live demo

https://match-intel.gentlecliff-a78733ad.canadacentral.azurecontainerapps.io
