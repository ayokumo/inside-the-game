"""Shared agent state. File JSON, versioned, with compare and swap so two agents cannot silently
overwrite each other. Swappable for an Azure database later behind the same three functions."""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "state"
_LOCK = threading.Lock()
_KEY = re.compile(r"^[A-Za-z0-9_\-:.]{1,80}$")


def _path(ns, key, root):
    if not _KEY.match(ns) or not _KEY.match(key) or ".." in ns + key:
        raise ValueError("bad namespace or key")
    return Path(root or ROOT) / ns.replace(":", "_") / (key.replace(":", "_") + ".json")


def state_get(ns, key, root=None):
    p = _path(ns, key, root)
    if not p.exists():
        return {"version": 0, "value": None}
    return json.loads(p.read_text())


def state_put(ns, key, value, expected_version=None, root=None):
    """Write value. If expected_version is given and stale, nothing is written and ok is False."""
    p = _path(ns, key, root)
    with _LOCK:
        cur = json.loads(p.read_text()) if p.exists() else {"version": 0, "value": None}
        if expected_version is not None and cur["version"] != expected_version:
            return {"ok": False, "version": cur["version"], "value": cur["value"], "reason": "version conflict"}
        new = {"version": cur["version"] + 1, "value": value}
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(new))
        os.replace(tmp, p)
        return {"ok": True, "version": new["version"]}
