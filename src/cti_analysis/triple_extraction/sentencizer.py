"""Document sentencizer for CTI pipeline.

Converts a document (PDF or plain text) into filtered sentence-level chunks.
This is the always-on first step of the extraction pipeline:

1. Extract per-page body text from docling structured output (if available)
2. Sentencize each page with spaCy
3. Filter boilerplate (cookies, menus, copyright, etc.)
4. Filter CTI-irrelevant sentences (no keywords or NER hits)
5. Return one Chunk per surviving sentence

Semantic chunking (optional, separate module) can then merge these sentence
chunks into larger coherent groups.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import List

from cti_analysis.models.documents import NormalizedDocument, Chunk, chunk_from_text

# =============================================================================
# spaCy (lazy-loaded)
# =============================================================================

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
        if "sentencizer" not in _nlp.pipe_names:
            _nlp.add_pipe("sentencizer")
    return _nlp


# =============================================================================
# Boilerplate filtering
# =============================================================================

_BOILERPLATE_PATTERNS = [
    r"\b(cookie|cookies|privacy|manage cookies|accept|reject)\b",
    r"\b(all microsoft|sign in|search the blog|more|solutions|products|services)\b",
    r"\bsubscribe\b|\bshare\b|\bread more\b|\bheader\b|\bfooter\b",
    r"^\s*[A-Z]{2,}(?:\s+[A-Z]{2,})*\s*$",           # ALL CAPS menus
    r"^\s*©\s*\d{4}\s",                               # copyright
    r"^\s*[\uE000-\uF8FF\uf000-\uf8ff].*$",          # icon fonts
]
_BP = [re.compile(p, re.I) for p in _BOILERPLATE_PATTERNS]


def is_boilerplate(sent: str) -> bool:
    s = sent.strip()
    if len(s) < 8:
        return True
    return any(rx.search(s) for rx in _BP)


# =============================================================================
# CTI relevance filtering
# =============================================================================

_CTI_KEYWORDS = {
    "apt", "malware", "c2", "command and control", "phishing", "ransomware",
    "exploit", "cve-", "lateral movement", "persistence", "beacon", "cobalt strike",
    "exfiltration", "loader", "backdoor", "webshell", "ioc", "infrastructure",
    "trojan", "botnet", "vulnerability", "threat", "attack", "intrusion",
    "campaign", "spear", "credential", "payload", "dropper",
}

_NER_RELEVANT_LABELS = {"ORG", "PRODUCT", "GPE", "DATE"}


def is_relevant(sent: str) -> bool:
    lower = sent.lower()
    if any(kw in lower for kw in _CTI_KEYWORDS):
        return True
    nlp = _get_nlp()
    doc = nlp(sent)
    labels = {ent.label_ for ent in doc.ents}
    return bool(labels & _NER_RELEVANT_LABELS)


# =============================================================================
# Docling page extraction
# =============================================================================

def extract_page_texts(docling_doc) -> List[tuple]:
    """Extract per-page body text from a docling document object.

    Returns list of (page_no, page_text) sorted by page number.
    """
    page_chunks = defaultdict(str)
    for ti in docling_doc.texts:
        if not getattr(ti, "prov", None):
            continue
        if hasattr(ti, "content_layer") and ti.content_layer != "body":
            continue
        try:
            p = ti.prov[0].page_no
            page_chunks[p] += " " + ti.text.strip()
        except Exception:
            continue
    return sorted(page_chunks.items())


# =============================================================================
# Main entry point
# =============================================================================

def sentencize_document(doc: NormalizedDocument) -> List[Chunk]:
    """Split a document into filtered sentence-level chunks.

    If the document has a docling_doc in its metadata (PDF source), extracts
    page-level text and processes per page. Otherwise falls back to treating
    the full text as a single page.

    Returns one Chunk per CTI-relevant, non-boilerplate sentence.
    """
    import logging
    logger = logging.getLogger(__name__)

    docling_doc = None
    if isinstance(doc.meta, dict):
        docling_doc = doc.meta.get("docling_doc")

    if docling_doc is not None:
        page_texts = extract_page_texts(docling_doc)
    else:
        page_texts = [(1, doc.text)]

    nlp = _get_nlp()
    chunks: List[Chunk] = []
    sent_idx = 0
    total_sents = 0
    total_boilerplate = 0
    total_irrelevant = 0

    for page_no, page_text in page_texts:
        spacy_doc = nlp(page_text)
        page_kept = 0
        for sent in spacy_doc.sents:
            text = sent.text.strip()
            if not text:
                continue
            total_sents += 1
            if is_boilerplate(text):
                total_boilerplate += 1
                continue
            if not is_relevant(text):
                total_irrelevant += 1
                continue
            chunks.append(chunk_from_text(
                doc,
                f"{doc.doc_id}_p{page_no}_s{sent_idx}",
                text,
                meta_override={"page_no": page_no, "sent_idx": sent_idx},
            ))
            sent_idx += 1
            page_kept += 1

    print(f"  [sentencize] {len(page_texts)} pages, {total_sents} sentences -> "
          f"{len(chunks)} kept ({total_boilerplate} boilerplate, {total_irrelevant} irrelevant filtered)",
          flush=True)

    return chunks
