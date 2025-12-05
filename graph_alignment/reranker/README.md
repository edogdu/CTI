# Graph Alignment — Cross‑Encoder Re‑ranker (ID & LABEL)

**Owner:** @Wittyusername12  
**Issue:** #5 “Graph Alignment: Implement Cross‑Encoder”  
**Folder:** `graph-alignment/reranker/`

## What this is
The re‑ranker reads each **query sentence** and a small set of **candidate ATT&CK IDs**,
scores them with a cross‑encoder, applies guardrails, and emits a **top‑1** pick plus
metrics (keep‑rate, flips, Precision@1). This is the “second‑pass picker” in the graph
alignment pipeline.

> Bill asked that each piece live in the repo with a short “what it is / inputs / outputs /
how to run” note and links in meeting notes; this folder is that note.

## Repo layout (this folder)
