# Submission text

## Project name
Match Intel (working title)

## Elevator pitch (paste into the form)
Match Intel is a multi agent football commentary system where every number has a receipt. AI commentary sounds
confident whether it is right or wrong, so our agents never write numbers. A Narrator proposes structured claims, code
recomputes each one from the raw event log, and the final sentence is rebuilt from verified values only. A Checker
rejects wrong drafts and the app shows them as "Caught and fixed". Persona agents (analyst, casual fan, player focus,
accessibility) and a translator change tone and language but cannot change a fact. Built with Microsoft Agent Framework
agents on a Microsoft Foundry model (gpt-4.1-mini), a custom MCP server, and hosted on Azure Container Apps. All data is
synthetic, produced by our own simulator with hidden causes we use to measure the system.

## Technologies used
Microsoft Agent Framework, Microsoft Foundry (gpt-4.1-mini), Model Context Protocol (MCP server and client),
Azure Container Apps, Azure Container Registry, GitHub.

## Video script (under 2 minutes, aim for 1:45)
0:00 to 0:12  Hook, on screen text: "AI commentary sounds confident even when it is wrong." Show a wrong looking line.
0:12 to 0:30  Open the app, press Next moment. A goal appears. "Every number here is a button."
0:30 to 0:50  Click a number. Receipt: writer claimed X, code recomputed Y, N events. "Agents never write numbers."
0:50 to 1:10  Click "Caught and fixed". Show the rejected draft and reason. "The checker caught a wrong claim and a
              banned word before you ever saw them."
1:10 to 1:30  Pick your club, switch to Accessibility, then German. "Same facts, different voice and language. Persona
              and translation agents cannot change a fact."
1:30 to 1:45  Architecture slide: Agent Framework agents, MCP, Foundry, Container Apps. State results from
              eval_results.md (rejected draft rate). One honest limit: the reviewer can miss things.
Record at 1080p, keep the cursor slow, no music with lyrics. Upload to YouTube as Public (not Unlisted).
