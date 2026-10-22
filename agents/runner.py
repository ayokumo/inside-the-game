"""Production runner: each role is its own Agent Framework agent. Analyst and Narrator get the match MCP server as a
tool (the same server the UI and any other client can use). Needs Foundry credentials, so it is the one part of the
system exercised only with a live model; the loop around it is tested with scripted agents."""
from __future__ import annotations

import json
import os
import re

from . import prompts

ROLE_PROMPTS = {"analyst": prompts.ANALYST, "narrator": prompts.NARRATOR, "checker": prompts.CHECKER,
                "persona": prompts.PERSONA, "translate": prompts.TRANSLATE}
MCP_ROLES = {"analyst", "narrator"}


def parse_json(text: str) -> dict:
    """Models sometimes wrap JSON in prose or code fences; take the outermost object."""
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in agent reply")
    return json.loads(m.group(0))


class AgentRunner:
    def __init__(self, mcp_url: str | None = None, model: str | None = None, endpoint: str | None = None):
        from agent_framework import Agent, MCPStreamableHTTPTool

        if os.environ.get("GITHUB_TOKEN") and not (endpoint or os.environ.get("FOUNDRY_PROJECT_ENDPOINT")):
            # development route: GitHub Models (OpenAI compatible). Same agents, different model host.
            from agent_framework_openai import OpenAIChatClient
            client = OpenAIChatClient(model=model or os.environ.get("MODEL", "openai/gpt-4.1-mini"),
                                      api_key=os.environ["GITHUB_TOKEN"], base_url="https://models.github.ai/inference")
            self.backend = "github-models"
        else:
            from agent_framework_foundry import FoundryChatClient
            from azure.identity import DefaultAzureCredential
            client = FoundryChatClient(project_endpoint=endpoint or os.environ["FOUNDRY_PROJECT_ENDPOINT"],
                                       model=model or os.environ["FOUNDRY_MODEL"], credential=DefaultAzureCredential())
            self.backend = "azure-foundry"
        self.mcp = MCPStreamableHTTPTool(name="match_intel", url=mcp_url or os.environ.get("MCP_URL", "http://127.0.0.1:8000/mcp"),
                                         approval_mode="never_require")
        self.agents = {role: Agent(client, instructions=text, name=role,
                                   tools=[self.mcp] if role in MCP_ROLES else None)
                       for role, text in ROLE_PROMPTS.items()}

    async def __aenter__(self):
        await self.mcp.__aenter__()
        return self

    async def __aexit__(self, *a):
        await self.mcp.__aexit__(*a)

    async def __call__(self, role: str, payload: dict) -> dict:
        last = None
        for _ in range(2):                                   # one retry on malformed JSON
            reply = await self.agents[role].run(json.dumps(payload, default=str))
            try:
                return parse_json(reply.text)
            except ValueError as e:
                last = e
                payload = dict(payload, format_error="reply was not a single JSON object")
        raise RuntimeError(f"{role} agent returned invalid JSON twice: {last}")
