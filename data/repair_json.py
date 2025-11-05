import json
import re
import pathlib

# Path to your ANNO-CTR test.json file
p = pathlib.Path("data/ANNO-CTR/AnnoCTR/ner_json/test.json")

# Read file (ignore encoding errors)
t = p.read_text(encoding="utf-8", errors="ignore").strip()
ok = None

# --- 1. Try normal JSON ---
try:
    x = json.loads(t)
    if isinstance(x, list):
        ok = x
    elif isinstance(x, dict) and "data" in x:
        ok = x["data"]
except Exception:
    pass

# --- 2. Try JSON Lines ---
if ok is None:
    arr = []
    good = True
    for line in t.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            arr.append(json.loads(line))
        except Exception:
            good = False
            break
    if good and arr:
        ok = arr

# --- 3. Try concatenated JSON objects ---
if ok is None:
    repaired = "[" + re.sub(r"}\s*{", "},{", t) + "]"
    try:
        ok = json.loads(repaired)
    except Exception:
        pass

# --- 4. Try extracting the first valid [ ... ] block ---
if ok is None:
    m = re.search(r"\[[\s\S]*\]", t)
    if m:
        try:
            ok = json.loads(m.group(0))
        except Exception:
            pass

# --- If all failed, stop ---
if ok is None:
    raise SystemExit("Could not repair JSON — please inspect manually.")

# --- Write repaired version ---
out = p.with_name(p.stem + "_repaired.json")
out.write_text(json.dumps(ok, ensure_ascii=False, indent=2), encoding="utf-8")
print("✅ Repaired JSON written to:", out)
