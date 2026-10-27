"""Delete cached moments that fell back to the template (and their views) so pregenerate.py redoes only those.
Usage: python3 scripts/purge_fallbacks.py"""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "data" / "cache"
gone = views = 0
for f in sorted((root / "base").glob("*.json")):
    if json.loads(f.read_text()).get("used_fallback"):
        key = f.stem                                   # e.g. SYN-0001__M3
        for v in (root / "views").glob(key + "__*.json"):
            v.unlink(); views += 1
        f.unlink(); gone += 1
print(f"removed {gone} fallback moments and {views} views; now run scripts/pregenerate.py")
