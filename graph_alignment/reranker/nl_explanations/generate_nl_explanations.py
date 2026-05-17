#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_nl_explanations.py
============================================================================
Deterministic natural-language explanation synthesis layer for the
CTI-HAL cross-encoder reranker (ACSAC 2026 submission).

Purpose
-------
Consume the per-query structured explanation record at `explanations.json`
and produce a single analyst-facing paragraph for each of the 146 test
queries, plus structured fields that trace every prose claim back to the
JSON record it came from.

Design framing
--------------
This is an "XAI Description" in the sense of Lyu, Apidianaki & Callison-Burch
(Computational Linguistics 50(2):657-723, 2024): a deterministic verbalization
of observable input-output signals. We do not claim faithfulness to the
internal reasoning of the MiniLM cross-encoder. We claim deterministic
synthesis of: per-token leave-one-out impact (observable model behaviour
under input ablation), BM25 lexical overlap (independent of the model),
ATT&CK hierarchy metadata (external ontology), ranked alternatives and
score gap (direct model outputs), and threat-actor attribution (metadata
provided by the upstream pipeline).

Citation anchors
----------------
- SPLAIN (Kazakova et al., arXiv:2311.11215, 2023): templates-as-contribution
  precedent in security XAI. "SPLAIN's template-based approach ensures
  consistent warning structure and vocabulary."
- AGIR (Perrina et al., IEEE BigData 2023): template-first CTI pipeline
  validated as an independent contribution.
- Lyu, Apidianaki & Callison-Burch (Comp. Ling. 50(2), 2024): plausibility
  vs. faithfulness; the language convention we use for scope-limited claims.
- Rastogi et al. (arXiv:2503.02065, 2025): N=248 survey + N=24 interviews;
  empirical anchor for 4-6 sentence length, translated vocabulary, and
  analyst-action-oriented closing language.
- Nadeem et al. SoK (EuroS&P 2023, pp. 221-240): 86% of security XAI papers
  lack user studies; this MVP follows that normative pattern.

Reproducibility
---------------
- Pure-Python deterministic generation, no LLM, no randomness.
- Same input file -> bit-identical output.
- All templates and design constants are in this single file.
- Input file SHA256 is recorded in provenance_nl.json.

Usage
-----
    python generate_nl_explanations.py \\
        --input ../explanations.json \\
        --output-dir .

Self-test mode runs internal tests and prints PASS/FAIL summary:
    python generate_nl_explanations.py --self-test
============================================================================
"""

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


# ============================================================================
# CONSTANTS AND LOOKUPS
# ============================================================================

TEMPLATE_VERSION = "v1.0"
SCHEMA_VERSION = "1"

# Canonical CTI literature spellings for the seven test-set actors.
# "Wizard Spider" is two words per CrowdStrike convention; OilRig is one word.
# The lowercase keys match the `actor` field in explanations.json.
ACTOR_DISPLAY = {
    "apt29":        "APT29",
    "carbanak":     "Carbanak",
    "fin6":         "FIN6",
    "fin7":         "FIN7",
    "oilrig":       "OilRig",
    "sandworm":     "Sandworm",
    "wizardspider": "Wizard Spider",
}

# Translate the categorical confidence label into a register suitable for
# analyst-facing prose. Per Rastogi et al. 2025, analysts respond to
# certainty language better than to "HIGH/MEDIUM/LOW" tier labels.
CONFIDENCE_TIER_TRANSLATED = {
    "HIGH":   "high",
    "MEDIUM": "moderate",
    "LOW":    "low",
}

# Words to lowercase inside an ALL-CAPS technique name when title-casing.
# This is only triggered on ICS techniques and similar all-caps source data.
TITLE_CASE_MINOR_WORDS = {
    "of", "the", "and", "in", "for", "with", "on", "at", "to",
    "a", "an", "or", "by", "as",
}

# Tokens that must NEVER appear in the prose output (jargon hygiene).
# A validation pass checks for these and fails if any appears in any
# explanation. See validate_explanation().
FORBIDDEN_JARGON_TOKENS = [
    "logit",         # use "confidence score" instead
    "logits",
    "softmax",
    "leave-one-out",
    "leave one out",
    "top-k",
    "top k",
    "top-K",
    "p@1",
    "p@k",
    "argmax",
    "loss",          # not training-loop language
    "embedding",
    "perturbation",
    "ablation",
]

# Design citations encoded into the provenance file so a reviewer can
# trace the design decisions to their literature anchors.
DESIGN_CITATIONS = {
    "templates_as_contribution":      "Kazakova et al., SPLAIN, arXiv:2311.11215, 2023",
    "templates_as_first_step":        "Perrina et al., AGIR, IEEE BigData 2023",
    "faithfulness_scope_language":    "Lyu, Apidianaki & Callison-Burch, Comp. Ling. 50(2), 2024",
    "analyst_attention_budget":       "Rastogi et al., arXiv:2503.02065, 2025",
    "absence_of_user_study_normative": "Nadeem et al. SoK, EuroS&P 2023, pp. 221-240",
    "xai_descriptions_vs_narratives": "Cambria et al. (Gemini report ref. 7)",
    "translated_vocabulary":          "Rastogi et al. 2025 + SANS SOC Survey 2024",
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def smart_title_case(s):
    """
    Title-case a string but keep minor words lowercase (except first/last).
    Used only for ICS techniques whose source text appears in ALL CAPS.

    >>> smart_title_case('MANIPULATION OF CONTROL')
    'Manipulation of Control'
    >>> smart_title_case('FOO')
    'Foo'
    """
    if not s:
        return s
    words = s.lower().split()
    if not words:
        return s
    result = []
    for i, w in enumerate(words):
        # Always capitalize first and last word; lowercase minor words in the middle.
        if i == 0 or i == len(words) - 1 or w not in TITLE_CASE_MINOR_WORDS:
            result.append(w.capitalize())
        else:
            result.append(w)
    return " ".join(result)


def resolve_name_from_text_preview(text_preview):
    """
    Parse the human-readable name from an alternative's text_preview field.

    Observed format (99.7% of cases):
        '{ID} \u2014 {Name}: {description}'
    Edge case for T0833 (ICS technique with no colon, all caps):
        'T0833 \u2014 MANIPULATION OF CONTROL'

    Returns the name string, or None if the format does not match.
    The em-dash is U+2014 with single spaces on either side.
    """
    if not text_preview:
        return None
    # Match: ID, optional whitespace, em-dash, whitespace, name, optional colon-or-end
    # The name group is non-greedy and stops at the first colon OR end-of-string.
    m = re.match(r'^[\w.]+\s*\u2014\s*([^:]+?)(?::|$)', text_preview)
    if not m:
        return None
    name = m.group(1).strip()
    if not name:
        return None
    # If name is uppercase (heuristic: more than 2 alpha chars and equals its upper form),
    # apply smart title case. This handles ICS technique names like 'MANIPULATION OF CONTROL'.
    alpha_chars = [c for c in name if c.isalpha()]
    if len(alpha_chars) > 2 and name == name.upper():
        name = smart_title_case(name)
    return name


def resolve_name(att_id, alternatives, technique_descriptions=None):
    """
    Look up a human-readable name for an ATT&CK ID by searching the
    alternatives list and parsing text_preview. Optional fallback to a
    technique_descriptions dict (which the MVP does not actually use, but
    we keep the hook for future extension).

    Returns the name string, or None if unresolvable.
    """
    if not att_id:
        return None
    for a in alternatives:
        if a.get('id') == att_id:
            name = resolve_name_from_text_preview(a.get('text_preview', ''))
            if name:
                return name
    # Optional fallback hook for technique_descriptions.json (not used in MVP)
    if technique_descriptions and att_id in technique_descriptions:
        desc = technique_descriptions[att_id]
        if isinstance(desc, dict):
            name = desc.get('name') or desc.get('short_name')
        else:
            name = str(desc)
        if name and name == name.upper() and len(name) > 2:
            name = smart_title_case(name)
        return name
    return None


def actor_display_name(actor):
    """
    Map the lowercase actor key to its canonical CTI display name.
    Falls back to simple title case for unknown actors.
    """
    if not actor:
        return "an unspecified threat actor"
    if actor in ACTOR_DISPLAY:
        return ACTOR_DISPLAY[actor]
    # Generic fallback for unknown actor strings.
    return actor.title()


def ordinal(n):
    """
    Convert a positive integer to its ordinal-suffix string.

    >>> ordinal(1)
    '1st'
    >>> ordinal(2)
    '2nd'
    >>> ordinal(11)
    '11th'
    >>> ordinal(21)
    '21st'
    """
    if n < 0:
        return str(n)
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    last_digit = n % 10
    suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(last_digit, 'th')
    return f"{n}{suffix}"


def english_join(items):
    """
    Join a list of strings into an English-language list.

    >>> english_join(['a'])
    "'a'"
    >>> english_join(['a', 'b'])
    "'a' and 'b'"
    >>> english_join(['a', 'b', 'c'])
    "'a', 'b', and 'c'"
    """
    quoted = [f"'{x}'" for x in items]
    if not quoted:
        return ""
    if len(quoted) == 1:
        return quoted[0]
    if len(quoted) == 2:
        return f"{quoted[0]} and {quoted[1]}"
    return ", ".join(quoted[:-1]) + ", and " + quoted[-1]


# ============================================================================
# DERIVE QUERY STATE
# ============================================================================
# Given a single raw record from explanations.json, compute every structured
# field the phrase builders need. This is the single point of truth for the
# matched-via taxonomy, the best-ranked-gold derivation for error cases, the
# top-N evidence token selection, and the parent-name resolution.

def select_evidence_tokens(token_importance, max_tokens=3):
    """
    Select the top evidence tokens from the LOO importance list.

    Strategy:
    1. Filter to impact='HIGH' tokens, sort by normalized descending.
    2. Deduplicate case-insensitively by cleaned surface form (a word may
       appear at multiple positions with HIGH impact; six queries in the
       dataset trigger this).
    3. If fewer than 2 distinct HIGH tokens result, supplement with
       impact='MEDIUM' tokens (also deduplicated against what we have).
    4. Cap total at max_tokens.

    Returns a list of cleaned token strings, in order of importance.
    Empty list if no HIGH or MEDIUM tokens are available.
    """
    high = [t for t in token_importance if t.get('impact') == 'HIGH']
    medium = [t for t in token_importance if t.get('impact') == 'MEDIUM']
    high.sort(key=lambda t: t.get('normalized', 0.0), reverse=True)
    # For MEDIUM we sort by absolute normalized so the highest-magnitude
    # tokens come first even if their sign is negative.
    medium.sort(key=lambda t: abs(t.get('normalized', 0.0)), reverse=True)

    cleaned = []
    seen = set()

    def _take_until_full(source):
        """Add cleaned tokens from `source` to `cleaned` until full or exhausted."""
        for t in source:
            if len(cleaned) >= max_tokens:
                return
            tok = t.get('token', '').strip(",.?!;:\"'()[]{}")
            if not tok:
                continue
            key = tok.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(tok)

    _take_until_full(high)
    if len(cleaned) < 2:
        # We didn't reach 2 distinct HIGH tokens after dedup; pull from MEDIUM
        # to fill up. This preserves the "show at least two evidence terms when
        # possible" design intent even when HIGH has duplicates.
        _take_until_full(medium)

    return cleaned


def derive_query_state(record, technique_descriptions=None):
    """
    Build the complete structured-field dict for a single query.

    The returned dict is the single point of truth for everything the phrase
    builders need. Every field here either comes directly from the JSON
    record or is derived from it via a documented rule.
    """
    query = record.get('query', '')
    query_norm = record.get('query_norm', query.lower())
    actor = record.get('actor', '')

    # ----- Prediction -----
    prediction = record.get('prediction', {})
    predicted_id = prediction.get('id', '')
    predicted_score = prediction.get('score', 0.0)
    is_correct = bool(prediction.get('correct', False))

    # ----- Primary gold (the single "headline" gold listed in the gold field) -----
    gold = record.get('gold', {})
    primary_gold_id = gold.get('id') if gold else None

    # ----- Confidence & margin -----
    confidence_tier = record.get('confidence', 'UNKNOWN')
    score_gap_to_next = record.get('margin', 0.0)

    # ----- Alternatives (sorted by rank already in the source data) -----
    alternatives = record.get('alternatives', [])

    # ----- All gold IDs in the top-5 (authoritative source: is_gold flags) -----
    # The `gold` field shows only the "primary" annotation; the alternatives'
    # is_gold flags reveal the full multi-gold set.
    all_gold_ids = [a['id'] for a in alternatives if a.get('is_gold')]
    n_gold_ids = len(all_gold_ids)
    is_multi_gold = n_gold_ids >= 2

    # ----- Matched-via taxonomy -----
    # primary_gold:     predicted matches the gold field exactly (the most common correct case)
    # alternative_gold: predicted is one of the golds, but not the primary one
    #                   (this is the "model picked a different valid label" case)
    # error:            predicted is not in any gold (the 8 miss cases)
    if not is_correct:
        matched_via = 'error'
    elif predicted_id == primary_gold_id:
        matched_via = 'primary_gold'
    elif predicted_id in all_gold_ids:
        matched_via = 'alternative_gold'
    else:
        # Defensive: 'correct' flag says True but predicted isn't in any gold list.
        # This shouldn't happen with valid data but we handle it gracefully.
        matched_via = 'primary_gold'

    # ----- Best-ranked gold (lowest rank number among is_gold alternatives) -----
    # For error cases this is the closest miss to report to the analyst.
    # For correct cases this is informational (often the predicted itself).
    best_gold_rank = None
    best_gold_id = None
    best_gold_score = None
    for a in alternatives:
        if a.get('is_gold'):
            r = a.get('rank', 99)
            if best_gold_rank is None or r < best_gold_rank:
                best_gold_rank = r
                best_gold_id = a['id']
                best_gold_score = a.get('score', 0.0)

    # ----- Names (via resolver) -----
    predicted_name = resolve_name(predicted_id, alternatives, technique_descriptions)
    primary_gold_name = resolve_name(primary_gold_id, alternatives, technique_descriptions)
    best_gold_name = resolve_name(best_gold_id, alternatives, technique_descriptions)

    # ----- Evidence tokens (LOO + BM25) -----
    token_importance = record.get('token_importance', [])
    top_evidence_tokens = select_evidence_tokens(token_importance, max_tokens=3)
    n_high_impact_tokens = sum(1 for t in token_importance if t.get('impact') == 'HIGH')
    n_medium_impact_tokens = sum(1 for t in token_importance if t.get('impact') == 'MEDIUM')

    bm25_overlap = list(record.get('bm25_overlap', []))
    # Filter out single-character noise tokens (e.g., 's' from possessive splits
    # like "victim's" -> "victim" + "s"). Real BM25 lexical signals are at least
    # 2 characters in this dataset; single-character tokens are tokenization
    # artifacts that degrade prose quality without adding information.
    bm25_overlap = [t for t in bm25_overlap if len(t) >= 2]
    # Deduplicate BM25 against LOO: don't repeat tokens that already appear in
    # the LOO evidence list. The comparison is case-insensitive and uses the
    # cleaned-token form (no surrounding punctuation).
    loo_lower = {t.lower() for t in top_evidence_tokens}
    bm25_extra = [t for t in bm25_overlap if t.lower() not in loo_lower]
    has_bm25_overlap = bool(bm25_overlap)
    has_bm25_extra = bool(bm25_extra)

    # ----- Hierarchy -----
    hierarchy = record.get('hierarchy', {})
    hierarchy_type = hierarchy.get('type', 'unknown')
    hierarchy_parent_id = hierarchy.get('parent')
    # NOTE: hierarchy.parent_name in the source data is always either null or
    # literally the parent ID (never a real name). We resolve the parent name
    # from the alternatives list instead, which works for ~22/24 sub-techniques.
    hierarchy_parent_name = None
    if hierarchy_parent_id:
        hierarchy_parent_name = resolve_name(
            hierarchy_parent_id, alternatives, technique_descriptions
        )
    hierarchy_n_children = len(hierarchy.get('children', []) or [])
    hierarchy_n_siblings = len(hierarchy.get('siblings', []) or [])

    # ----- Next-ranked alternative (rank 2) -----
    next_alt = None
    for a in alternatives:
        if a.get('rank') == 2:
            next_alt = a
            break
    if next_alt is not None:
        next_alt_id = next_alt.get('id')
        next_alt_score = next_alt.get('score')
        next_alt_is_gold = bool(next_alt.get('is_gold', False))
        next_alt_name = resolve_name(next_alt_id, alternatives, technique_descriptions)
    else:
        next_alt_id = None
        next_alt_score = None
        next_alt_is_gold = False
        next_alt_name = None

    # ----- Other golds (besides predicted) -- for multi-gold framing -----
    other_gold_ids = [g for g in all_gold_ids if g != predicted_id]
    other_gold_names = []
    for gid in other_gold_ids[:3]:  # cap at 3 for prose readability
        name = resolve_name(gid, alternatives, technique_descriptions)
        other_gold_names.append({'id': gid, 'name': name})

    return {
        # ---- Identity ----
        'query':                       query,
        'query_norm':                  query_norm,
        'actor':                       actor,
        'actor_display':               actor_display_name(actor),

        # ---- Prediction ----
        'predicted_id':                predicted_id,
        'predicted_name':              predicted_name,
        'predicted_score':             predicted_score,
        'confidence_tier':             confidence_tier,
        'confidence_tier_translated':  CONFIDENCE_TIER_TRANSLATED.get(
                                           confidence_tier, confidence_tier.lower()),
        'score_gap_to_next':           score_gap_to_next,

        # ---- Correctness ----
        'is_correct':                  is_correct,
        'primary_gold_id':             primary_gold_id,
        'primary_gold_name':           primary_gold_name,
        'all_gold_ids':                all_gold_ids,
        'n_gold_ids':                  n_gold_ids,
        'is_multi_gold':               is_multi_gold,
        'matched_via':                 matched_via,
        'best_gold_id':                best_gold_id,
        'best_gold_name':              best_gold_name,
        'best_gold_rank':              best_gold_rank,
        'best_gold_score':             best_gold_score,
        'other_golds':                 other_gold_names,

        # ---- Evidence ----
        'top_evidence_tokens':         top_evidence_tokens,
        'n_high_impact_tokens':        n_high_impact_tokens,
        'n_medium_impact_tokens':      n_medium_impact_tokens,
        'has_bm25_overlap':            has_bm25_overlap,
        'bm25_overlap_tokens':         bm25_overlap,
        'bm25_extra_tokens':           bm25_extra,
        'has_bm25_extra':              has_bm25_extra,

        # ---- Hierarchy ----
        'hierarchy_type':              hierarchy_type,
        'hierarchy_parent_id':         hierarchy_parent_id,
        'hierarchy_parent_name':       hierarchy_parent_name,
        'hierarchy_n_children':        hierarchy_n_children,
        'hierarchy_n_siblings':        hierarchy_n_siblings,

        # ---- Next alternative ----
        'next_alt_id':                 next_alt_id,
        'next_alt_name':               next_alt_name,
        'next_alt_score':              next_alt_score,
        'next_alt_is_gold':            next_alt_is_gold,
    }


# ============================================================================
# PHRASE BUILDERS (one per sentence slot in the final paragraph)
# ============================================================================
# Order follows the Gemini-recommended sequence: deployment context first,
# primary prediction next, evidence, correctness framing, then analyst action.

def _format_id_with_name(att_id, name):
    """
    Render an ATT&CK ID with its name in bold-markdown form, falling back
    to just the bold ID when the name failed to resolve.

    Example: 'T1059.001', 'PowerShell' -> '**T1059.001 (PowerShell)**'
    Example: 'T9999', None              -> '**T9999**'
    """
    if not att_id:
        return ""
    if name:
        return f"**{att_id} ({name})**"
    return f"**{att_id}**"


def build_s1_deployment_and_prediction(state):
    """
    Sentence 1: Deployment context (actor) + primary prediction + confidence + score gap.

    Grammar: the actor opens with a prepositional clause that grammatically
    subordinates it (the model is the active subject; the actor is passive
    context per the Step D counterfactual probe finding). The translated
    confidence vocabulary follows Rastogi et al.'s analyst-facing register.
    """
    actor = state['actor_display']
    pred = _format_id_with_name(state['predicted_id'], state['predicted_name'])
    tier = state['confidence_tier_translated']
    score = state['predicted_score']
    gap = state['score_gap_to_next']
    return (
        f"In a report attributed to {actor}, the reranker assigned {pred} "
        f"as its top-ranked candidate with {tier} certainty "
        f"(confidence score {score:.2f}, score gap of {gap:.2f} over the next candidate)."
    )


def build_s2_evidence(state):
    """
    Sentence 2: Token-level and lexical-overlap evidence.

    Branching:
      A. Standard case (>=2 LOO HIGH/MEDIUM tokens): name 2-3 key tokens, append BM25 if present.
      B. One LOO token: name the one token, append BM25 if present.
      C. No LOO tokens but BM25 present: BM25 is the principal evidence.
      D. No LOO tokens and no BM25 (very short queries): direct-match fallback.

    The wording avoids "leave-one-out," "logit," "margin" per the
    translated-vocabulary convention.
    """
    loo_tokens = state['top_evidence_tokens']
    bm25_extra = state['bm25_extra_tokens']
    has_bm25_extra = state['has_bm25_extra']

    if loo_tokens:
        # Branch A or B: LOO is available.
        evidence = english_join(loo_tokens)
        if len(loo_tokens) == 1:
            base = f"The strongest token-level evidence is the term {evidence}"
        else:
            base = f"The strongest token-level evidence comes from the terms {evidence}"
        if has_bm25_extra:
            overlap = english_join(bm25_extra[:3])  # cap at 3 to avoid sentence sprawl
            return f"{base}, with additional lexical overlap against the candidate description on {overlap}."
        return f"{base}."

    # Branch C or D: no LOO evidence.
    if has_bm25_extra or state['has_bm25_overlap']:
        # Use whichever BM25 list is non-empty (extra is empty when overlap was fully de-duped,
        # but for no-LOO case we just want the raw overlap).
        bm25_for_prose = bm25_extra if bm25_extra else state['bm25_overlap_tokens']
        overlap = english_join(bm25_for_prose[:3])
        return (
            f"With a short query, individual token-impact attribution does not apply; "
            f"the principal evidence is lexical overlap with the candidate description "
            f"on {overlap}."
        )

    # Branch D: nothing to point at. State this honestly.
    return (
        "With a single-token query and no lexical overlap against the candidate description, "
        "individual evidence attribution does not apply; the model matched the query token "
        "directly against the candidate entry."
    )


def build_s3_correctness(state):
    """
    Sentence 3: Multi-gold / single-gold / error framing.

    Per all four research reports, the prose uses "annotated" rather than
    "correct" or "ground truth" to respect the multi-label evaluation
    convention (93% of queries have >=2 valid mappings).
    """
    matched_via = state['matched_via']
    pred_id_bold = f"**{state['predicted_id']}**"
    n_golds = state['n_gold_ids']

    if matched_via == 'primary_gold':
        if state['is_multi_gold']:
            return (
                f"{pred_id_bold} is one of {n_golds} ATT&CK techniques annotated "
                f"as a valid mapping for this query."
            )
        return (
            f"{pred_id_bold} is the single ATT&CK technique annotated for this query."
        )

    if matched_via == 'alternative_gold':
        # The model picked a non-primary gold from a multi-gold set.
        # Edge case: n_golds == 1 means our top-5 surfaces exactly one gold and
        # it's the predicted, while the `gold` field designates a different
        # "primary" technique that doesn't appear in our top-5 alternatives.
        # The phrase "one of 1 techniques" is grammatically wrong, so we
        # special-case it (1 of 146 queries in the dataset).
        if n_golds == 1:
            return (
                f"{pred_id_bold} matches an annotated valid mapping for this query, "
                f"though the dataset's primary annotation does not appear in the "
                f"top-5 candidates."
            )
        # If the primary annotation's name can't be resolved (10/146 cases where
        # the primary gold isn't in the top-5 alternatives), omit the dangling-
        # name clause instead of showing a bare ID like "**T1213.002**" that
        # the reader has no context for.
        if state['primary_gold_name']:
            primary = _format_id_with_name(
                state['primary_gold_id'], state['primary_gold_name']
            )
            return (
                f"{pred_id_bold} is one of {n_golds} ATT&CK techniques annotated as a valid "
                f"mapping for this query; the primary annotation is {primary}."
            )
        return (
            f"{pred_id_bold} is one of {n_golds} ATT&CK techniques annotated "
            f"as a valid mapping for this query."
        )

    # Error case
    best = _format_id_with_name(state['best_gold_id'], state['best_gold_name'])
    rank_word = ordinal(state['best_gold_rank']) if state['best_gold_rank'] else "lower"
    score_part = (
        f" with confidence score {state['best_gold_score']:.2f}"
        if state['best_gold_score'] is not None else ""
    )
    return (
        f"However, the annotated technique {best} ranked {rank_word} "
        f"in the model's candidate list{score_part}."
    )


def build_s4_alternative_and_hierarchy(state):
    """
    Sentence 4: Runner-up alternative + ATT&CK hierarchy context.

    For correct cases: name the rank-2 candidate, then fold in hierarchy.
    For error cases: hierarchy only (the rank-2 candidate is already mentioned
    in S3 as the best-ranked gold), to avoid redundancy.
    """
    htype = state['hierarchy_type']
    pred_id = state['predicted_id']

    # Build the hierarchy clause based on the four types.
    if htype == 'tactic':
        hierarchy_clause = f"{pred_id} is an ATT&CK tactic, a high-level adversary objective"
    elif htype == 'technique':
        hierarchy_clause = f"{pred_id} is a top-level ATT&CK technique"
    elif htype == 'sub-technique':
        parent = state['hierarchy_parent_id']
        parent_name = state['hierarchy_parent_name']
        if parent and parent_name:
            hierarchy_clause = (
                f"{pred_id} is a sub-technique of **{parent} ({parent_name})**"
            )
        elif parent:
            hierarchy_clause = f"{pred_id} is a sub-technique of **{parent}**"
        else:
            hierarchy_clause = f"{pred_id} is an ATT&CK sub-technique"
    elif htype == 'software':
        hierarchy_clause = f"{pred_id} is an ATT&CK software entry"
    else:
        hierarchy_clause = f"{pred_id} sits in the ATT&CK framework"

    if state['matched_via'] == 'error':
        # The rank-2 candidate is the annotated gold mentioned in S3.
        # S4 carries only the hierarchy fact, so the analyst sees where the
        # model's prediction sits structurally.
        return f"In structural terms, {hierarchy_clause}."

    # Avoid double-mentioning the same technique across S3 and S4:
    # in alternative_gold cases, the rank-2 candidate is often the same
    # technique already named as "the primary annotation" in S3. Detect
    # that exact case and elide the runner-up clause.
    skip_runner_up = (
        state['matched_via'] == 'alternative_gold'
        and state['next_alt_id'] == state['primary_gold_id']
    )

    if skip_runner_up or not state['next_alt_id']:
        return f"In structural terms, {hierarchy_clause}."

    # Correct case with a distinct runner-up: name it and add hierarchy.
    next_alt = _format_id_with_name(state['next_alt_id'], state['next_alt_name'])
    next_score = state['next_alt_score']
    score_part = (
        f" at confidence score {next_score:.2f}"
        if next_score is not None else ""
    )
    return (
        f"The next-ranked candidate was {next_alt}{score_part}; "
        f"in structural terms, {hierarchy_clause}."
    )


def build_s5_analyst_action(state):
    """
    Sentence 5: Analyst-action implication, scaled to confidence tier and correctness.

    Per Rastogi et al. 2025, Tier-1 analysts say "What I need most are
    straightforward next steps." This sentence translates the confidence
    tier and correctness state into a recommended workflow disposition.
    """
    tier = state['confidence_tier']  # raw HIGH/MEDIUM/LOW
    is_correct = state['is_correct']
    matched_via = state['matched_via']

    if matched_via != 'error':
        # Correct cases (primary_gold or alternative_gold).
        if tier == 'HIGH':
            return (
                "This is a high-certainty assignment suitable for downstream automation "
                "with light spot-checking."
            )
        if tier == 'MEDIUM':
            return (
                "This is a moderate-certainty assignment; brief analyst review is "
                "recommended before downstream automation."
            )
        # LOW
        return (
            "Analyst review is recommended before downstream use given the low certainty."
        )

    # Error cases: model disagrees with the annotated mapping.
    if tier == 'HIGH':
        return (
            "Despite the model's high certainty, analyst review is required to reconcile "
            "the disagreement with the annotated mapping."
        )
    if tier == 'MEDIUM':
        return (
            "Given the moderate certainty and the small score gap, analyst review is "
            "recommended to reconcile the disagreement with the annotated mapping."
        )
    # LOW
    return (
        "Analyst review is required given the low certainty and the disagreement with "
        "the annotated mapping."
    )


# ============================================================================
# ORCHESTRATOR AND VALIDATOR
# ============================================================================

def compose_explanation(state):
    """
    Compose the five sentences into a single paragraph.

    Returns the prose string. The structured fields are returned separately
    by the caller alongside the prose.
    """
    s1 = build_s1_deployment_and_prediction(state)
    s2 = build_s2_evidence(state)
    s3 = build_s3_correctness(state)
    s4 = build_s4_alternative_and_hierarchy(state)
    s5 = build_s5_analyst_action(state)
    return " ".join([s1, s2, s3, s4, s5])


def _count_sentences(text):
    """
    Count sentence-final periods. Periods inside numbers (e.g., '4.66') are
    not followed by a space and do not match the ". " sentence separator.
    """
    if not text:
        return 0
    # Sentences are separated by ". " (with the final sentence terminated by
    # just "."). Counting separators and adding 1 gives the sentence total.
    # Note: we also match ".\n" defensively, but we never produce that.
    n_separators = text.count(". ")
    # The final sentence ends with "." but no following space.
    if text.rstrip().endswith("."):
        return n_separators + 1
    return n_separators


def validate_explanation(prose, state, min_sentences=5, max_sentences=5,
                          min_words=70, max_words=180):
    """
    Validate that an explanation paragraph meets the structural requirements
    we promised in the design. Returns a list of error strings; an empty list
    indicates the explanation passes all checks.

    Checks performed:
      1. Sentence count is exactly 5 (the locked-in MVP shape).
      2. Word count is between min_words and max_words.
      3. The predicted ATT&CK ID appears in the prose.
      4. The actor display name appears in the prose.
      5. A confidence-tier word ('high'/'moderate'/'low') appears.
      6. The token 'annotat' appears (matches 'annotated' or 'annotation';
         catches both the multi-gold framing and the error-case framing).
      7. None of the forbidden jargon tokens appear (case-insensitive).
    """
    errors = []
    if not prose:
        errors.append("prose is empty")
        return errors

    # 1. Sentence count
    n_sent = _count_sentences(prose)
    if not (min_sentences <= n_sent <= max_sentences):
        errors.append(
            f"sentence count {n_sent} outside [{min_sentences}, {max_sentences}]"
        )

    # 2. Word count
    n_words = len(prose.split())
    if not (min_words <= n_words <= max_words):
        errors.append(
            f"word count {n_words} outside [{min_words}, {max_words}]"
        )

    # 3. Predicted ID must appear (this catches templating bugs)
    if state['predicted_id'] not in prose:
        errors.append(f"predicted_id {state['predicted_id']!r} not found in prose")

    # 4. Actor display name must appear
    if state['actor_display'] not in prose:
        errors.append(f"actor_display {state['actor_display']!r} not found in prose")

    # 5. Confidence tier word must appear
    tier_word = state['confidence_tier_translated']
    if tier_word not in prose.lower():
        errors.append(f"confidence tier word {tier_word!r} not found")

    # 6. Multi-gold / error framing must use 'annotat'
    if 'annotat' not in prose.lower():
        errors.append("multi-gold/error framing ('annotated'/'annotation') not found")

    # 7. Forbidden jargon must NOT appear
    prose_lower = prose.lower()
    for tok in FORBIDDEN_JARGON_TOKENS:
        if tok.lower() in prose_lower:
            errors.append(f"forbidden jargon token {tok!r} found in prose")

    return errors


# ============================================================================
# RECORD BUILDER (combines state + prose into the output record)
# ============================================================================

def build_output_record(state, prose):
    """
    Assemble the output JSON record: the prose explanation plus every
    structured field that traces back to a JSON field in the source.
    """
    return {
        # Identity
        'query':                       state['query'],
        'query_norm':                  state['query_norm'],
        'actor':                       state['actor'],
        'actor_display':               state['actor_display'],

        # Prediction
        'predicted_id':                state['predicted_id'],
        'predicted_name':              state['predicted_name'],
        'predicted_score':             state['predicted_score'],
        'confidence_tier':             state['confidence_tier'],
        'confidence_tier_translated':  state['confidence_tier_translated'],
        'score_gap_to_next':           state['score_gap_to_next'],

        # Correctness
        'is_correct':                  state['is_correct'],
        'primary_gold_id':             state['primary_gold_id'],
        'primary_gold_name':           state['primary_gold_name'],
        'all_gold_ids':                state['all_gold_ids'],
        'n_gold_ids':                  state['n_gold_ids'],
        'is_multi_gold':               state['is_multi_gold'],
        'matched_via':                 state['matched_via'],
        'best_gold_id':                state['best_gold_id'],
        'best_gold_name':              state['best_gold_name'],
        'best_gold_rank':              state['best_gold_rank'],
        'best_gold_score':             state['best_gold_score'],
        'other_golds':                 state['other_golds'],

        # Evidence
        'top_evidence_tokens':         state['top_evidence_tokens'],
        'n_high_impact_tokens':        state['n_high_impact_tokens'],
        'n_medium_impact_tokens':      state['n_medium_impact_tokens'],
        'has_bm25_overlap':            state['has_bm25_overlap'],
        'bm25_overlap_tokens':         state['bm25_overlap_tokens'],
        'bm25_extra_tokens':           state['bm25_extra_tokens'],

        # Hierarchy
        'hierarchy_type':              state['hierarchy_type'],
        'hierarchy_parent_id':         state['hierarchy_parent_id'],
        'hierarchy_parent_name':       state['hierarchy_parent_name'],
        'hierarchy_n_children':        state['hierarchy_n_children'],
        'hierarchy_n_siblings':        state['hierarchy_n_siblings'],

        # Next alternative
        'next_alt_id':                 state['next_alt_id'],
        'next_alt_name':               state['next_alt_name'],
        'next_alt_score':              state['next_alt_score'],
        'next_alt_is_gold':            state['next_alt_is_gold'],

        # Prose
        'explanation':                 prose,

        # Structural meta
        'n_sentences':                 _count_sentences(prose),
        'n_words':                     len(prose.split()),
    }


# ============================================================================
# I/O AND MAIN
# ============================================================================

def sha256_of_file(path):
    """Compute the SHA-256 hex digest of a file's bytes."""
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def load_explanations(input_path):
    """Load the explanations.json input file with UTF-8 encoding."""
    with open(input_path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def write_json(path, data):
    """
    Write a JSON file with UTF-8 encoding, sorted keys, 2-space indent.

    The `newline=''` argument disables Python's platform-default newline
    translation so that the file contains LF-only line endings on every
    platform, including Windows where the default would otherwise convert
    every '\\n' to '\\r\\n' and produce a different byte sequence (and
    therefore a different SHA-256 hash) than the same script on Linux or
    macOS. Cross-platform byte-identical output is required for the
    reproducibility claim in the paper.
    """
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        json.dump(data, fh, indent=2, sort_keys=False, ensure_ascii=False)


def write_text(path, text):
    """
    Write a text file with UTF-8 encoding.

    See write_json above for the rationale on `newline=''`. Same reason:
    cross-platform byte-identical output requires disabling Python's
    default newline translation.
    """
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        fh.write(text)


def build_provenance(input_path, n_records, template_version):
    """Build the provenance record for reproducibility tracking."""
    return {
        'generated_at_utc':   datetime.now(timezone.utc).isoformat(),
        'script':             os.path.basename(__file__),
        'template_version':   template_version,
        'schema_version':     SCHEMA_VERSION,
        'input_path':         os.path.abspath(input_path),
        'input_sha256':       sha256_of_file(input_path),
        'n_records':          n_records,
        'design_citations':   DESIGN_CITATIONS,
        'python_version':     sys.version.split()[0],
    }


def build_template_versions_artifact():
    """
    Snapshot the template strings into a separate file so reviewers can
    audit the templates without reading the source code.
    """
    return {
        'template_version': TEMPLATE_VERSION,
        'schema_version': SCHEMA_VERSION,
        'sentence_order': [
            'S1: deployment context + primary prediction + confidence + score gap',
            'S2: token + lexical evidence (or short-query fallback)',
            'S3: multi-gold / single-gold / error correctness framing',
            'S4: next-ranked alternative (when distinct from primary gold) + hierarchy',
            'S5: analyst-action implication scaled to confidence tier and correctness',
        ],
        'vocabulary_translations': CONFIDENCE_TIER_TRANSLATED,
        'actor_display_map': ACTOR_DISPLAY,
        'forbidden_jargon_tokens': FORBIDDEN_JARGON_TOKENS,
        'design_citations': DESIGN_CITATIONS,
        'notes': [
            "Templates are written as deterministic f-strings inside the phrase builders.",
            "No randomness; same input -> bit-identical output across runs.",
            "Bold markers (**X**) are markdown for LaTeX-rendered paper figures.",
            "Em-dash U+2014 is used in alternative text_preview; the resolver handles it.",
        ],
    }


# ============================================================================
# CURATED EXAMPLES FOR PAPER
# ============================================================================

def curate_paper_examples(records):
    """
    Select 5 representative examples for inclusion in the paper figure box.
    Selection criteria, deliberately covering the case space:
      1. HIGH-confidence single-gold correct (the easy case)
      2. HIGH-confidence multi-gold alternative_gold match (the typical case)
      3. MEDIUM-confidence correct (showing hedged language)
      4. Error case where gold ranked 2nd (the typical error pattern)
      5. Software/short-query case (showing BM25-only evidence fallback)
    """
    examples = []

    # 1. HIGH-confidence single-gold correct, longest score gap (clearest case)
    candidates = [
        r for r in records
        if r['confidence_tier'] == 'HIGH'
        and r['matched_via'] == 'primary_gold'
        and not r['is_multi_gold']
    ]
    if candidates:
        candidates.sort(key=lambda r: r['score_gap_to_next'], reverse=True)
        examples.append({
            'label': 'HIGH confidence, single-gold, primary_gold match',
            'reason': 'the easy case: clean single-label match with large score gap',
            'record': candidates[0],
        })

    # 2. HIGH-confidence multi-gold alternative_gold match (the typical case)
    candidates = [
        r for r in records
        if r['confidence_tier'] == 'HIGH'
        and r['matched_via'] == 'alternative_gold'
        and r['is_multi_gold']
    ]
    if candidates:
        candidates.sort(key=lambda r: r['score_gap_to_next'], reverse=True)
        examples.append({
            'label': 'HIGH confidence, multi-gold, alternative_gold match',
            'reason': 'the dominant case: model picked one of multiple valid annotations',
            'record': candidates[0],
        })

    # 3. MEDIUM-confidence correct
    candidates = [
        r for r in records
        if r['confidence_tier'] == 'MEDIUM' and r['is_correct']
    ]
    if candidates:
        # Pick one with a moderate score gap to show hedged language
        candidates.sort(key=lambda r: abs(r['score_gap_to_next'] - 1.0))
        examples.append({
            'label': 'MEDIUM confidence, correct',
            'reason': 'shows hedged language and brief-review recommendation',
            'record': candidates[0],
        })

    # 4. Error case where gold ranked 2nd
    candidates = [
        r for r in records
        if r['matched_via'] == 'error'
        and r['best_gold_rank'] == 2
    ]
    if candidates:
        # Prefer MEDIUM-confidence error with small gap (the "close call" pattern)
        candidates.sort(key=lambda r: (
            0 if r['confidence_tier'] == 'MEDIUM' else 1,
            r['score_gap_to_next']
        ))
        examples.append({
            'label': 'Error case, gold at rank 2 (the typical error pattern)',
            'reason': 'shows error framing with the annotated technique nearby',
            'record': candidates[0],
        })

    # 5. Software / short-query case with BM25-only evidence
    candidates = [
        r for r in records
        if r['hierarchy_type'] == 'software'
        and not r['top_evidence_tokens']
        and r['has_bm25_overlap']
    ]
    if candidates:
        examples.append({
            'label': 'Software / short-query, BM25-only evidence',
            'reason': 'shows the no-LOO evidence fallback path',
            'record': candidates[0],
        })

    return examples


def render_examples_markdown(curated):
    """
    Render the curated paper examples as a markdown file suitable for
    inclusion in the paper figure box or appendix listing.
    """
    lines = []
    lines.append("# Curated Natural-Language Explanation Examples")
    lines.append("")
    lines.append("These examples are selected to illustrate the explanation pattern across")
    lines.append("the case space: confidence tiers, multi-gold cases, error cases, and the")
    lines.append("short-query fallback. Complete explanations for all 146 test queries are")
    lines.append("in `nl_explanations.json`.")
    lines.append("")

    for i, ex in enumerate(curated, start=1):
        rec = ex['record']
        lines.append(f"## Example {i}: {ex['label']}")
        lines.append("")
        lines.append(f"*Why this example:* {ex['reason']}.")
        lines.append("")
        lines.append("**Query (verbatim from source CTI report):**")
        lines.append("")
        lines.append("> " + rec['query'])
        lines.append("")
        lines.append("**Structured signals consumed by the template:**")
        lines.append("")
        lines.append(f"- Actor (metadata, not model input): `{rec['actor']}` -> `{rec['actor_display']}`")
        lines.append(f"- Predicted: `{rec['predicted_id']}` ({rec['predicted_name']})")
        lines.append(f"- Confidence: `{rec['confidence_tier']}` (translated to `{rec['confidence_tier_translated']}`)")
        lines.append(f"- Confidence score: `{rec['predicted_score']:.4f}`")
        lines.append(f"- Score gap to next: `{rec['score_gap_to_next']:.4f}`")
        lines.append(f"- Matched via: `{rec['matched_via']}`")
        lines.append(f"- All gold IDs (n={rec['n_gold_ids']}): `{rec['all_gold_ids']}`")
        lines.append(f"- Top LOO evidence tokens: `{rec['top_evidence_tokens']}`")
        lines.append(f"- BM25 overlap (post-dedup): `{rec['bm25_extra_tokens']}`")
        lines.append(f"- Hierarchy: `{rec['hierarchy_type']}`"
                     + (f", parent `{rec['hierarchy_parent_id']}`"
                        + (f" ({rec['hierarchy_parent_name']})" if rec['hierarchy_parent_name'] else "")
                        if rec['hierarchy_parent_id'] else ""))
        lines.append("")
        lines.append("**Generated explanation paragraph:**")
        lines.append("")
        lines.append(rec['explanation'])
        lines.append("")
        lines.append(f"*({rec['n_sentences']} sentences, {rec['n_words']} words)*")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ============================================================================
# SELF-TEST MODE
# ============================================================================

def run_self_tests():
    """
    Internal sanity-check suite. Returns (n_pass, n_fail, failure_messages).
    Exercises each helper and phrase builder with synthetic and real data.
    """
    failures = []
    n_total = 0

    def check(label, condition, expected=None, actual=None):
        nonlocal n_total
        n_total += 1
        if not condition:
            msg = f"FAIL: {label}"
            if expected is not None or actual is not None:
                msg += f" (expected {expected!r}, got {actual!r})"
            failures.append(msg)

    # ---- smart_title_case ----
    check("smart_title_case MANIPULATION OF CONTROL",
          smart_title_case('MANIPULATION OF CONTROL') == 'Manipulation of Control')
    check("smart_title_case FOO", smart_title_case('FOO') == 'Foo')
    check("smart_title_case empty", smart_title_case('') == '')

    # ---- resolve_name_from_text_preview ----
    check("resolver standard format",
          resolve_name_from_text_preview('T1518.001 \u2014 Security Software Discovery: Adversaries may attempt...')
          == 'Security Software Discovery')
    check("resolver software with brackets",
          resolve_name_from_text_preview('S0266 \u2014 TrickBot: [TrickBot](https://...)')
          == 'TrickBot')
    check("resolver T0833 (no colon, all caps)",
          resolve_name_from_text_preview('T0833 \u2014 MANIPULATION OF CONTROL')
          == 'Manipulation of Control')
    check("resolver None input",
          resolve_name_from_text_preview(None) is None)
    check("resolver empty input",
          resolve_name_from_text_preview('') is None)
    check("resolver no em-dash",
          resolve_name_from_text_preview('T1234 no em-dash here') is None)

    # ---- actor_display_name ----
    check("actor apt29", actor_display_name('apt29') == 'APT29')
    check("actor wizardspider two words",
          actor_display_name('wizardspider') == 'Wizard Spider')
    check("actor oilrig one word",
          actor_display_name('oilrig') == 'OilRig')
    check("actor empty", actor_display_name('') == 'an unspecified threat actor')
    check("actor unknown", actor_display_name('unknownactor') == 'Unknownactor')

    # ---- ordinal ----
    for n, exp in [(1, '1st'), (2, '2nd'), (3, '3rd'), (4, '4th'),
                   (11, '11th'), (21, '21st'), (102, '102nd')]:
        check(f"ordinal({n})", ordinal(n) == exp, exp, ordinal(n))

    # ---- english_join ----
    check("english_join 1", english_join(['a']) == "'a'")
    check("english_join 2", english_join(['a', 'b']) == "'a' and 'b'")
    check("english_join 3", english_join(['a', 'b', 'c']) == "'a', 'b', and 'c'")

    # ---- select_evidence_tokens ----
    # Design intent (from docstring): if we have >=2 HIGH tokens, don't dilute
    # with weaker MEDIUM. Only supplement with MEDIUM when HIGH count is < 2.
    toks_two_high = [
        {'token': 'foo', 'normalized': 1.0, 'impact': 'HIGH'},
        {'token': 'bar', 'normalized': 0.6, 'impact': 'HIGH'},
        {'token': 'baz', 'normalized': 0.3, 'impact': 'MEDIUM'},
        {'token': 'qux', 'normalized': 0.1, 'impact': 'LOW'},
        {'token': 'noise', 'normalized': 0.0, 'impact': 'NONE'},
    ]
    sel = select_evidence_tokens(toks_two_high, max_tokens=3)
    check("evidence selection: 2 HIGH available -> use only HIGH (no dilution)",
          sel == ['foo', 'bar'], ['foo', 'bar'], sel)

    # When we have only 1 HIGH, supplement with MEDIUM up to max_tokens.
    toks_one_high = [
        {'token': 'foo', 'normalized': 1.0, 'impact': 'HIGH'},
        {'token': 'bar', 'normalized': 0.5, 'impact': 'MEDIUM'},
        {'token': 'baz', 'normalized': 0.3, 'impact': 'MEDIUM'},
        {'token': 'qux', 'normalized': 0.1, 'impact': 'LOW'},
    ]
    sel = select_evidence_tokens(toks_one_high, max_tokens=3)
    check("evidence selection: 1 HIGH -> supplement with MEDIUM up to max_tokens",
          sel == ['foo', 'bar', 'baz'], ['foo', 'bar', 'baz'], sel)

    # When we have 0 HIGH, use MEDIUM only.
    toks_no_high = [
        {'token': 'bar', 'normalized': 0.5, 'impact': 'MEDIUM'},
        {'token': 'baz', 'normalized': 0.3, 'impact': 'MEDIUM'},
        {'token': 'qux', 'normalized': 0.1, 'impact': 'LOW'},
    ]
    sel = select_evidence_tokens(toks_no_high, max_tokens=3)
    check("evidence selection: 0 HIGH -> fall back to MEDIUM",
          sel == ['bar', 'baz'], ['bar', 'baz'], sel)

    # Trailing-punctuation strip
    toks_punct = [{'token': 'Trickbot,', 'normalized': 1.0, 'impact': 'HIGH'}]
    sel2 = select_evidence_tokens(toks_punct)
    check("evidence selection strips trailing comma", sel2 == ['Trickbot'])

    # ---- Determinism + Validator (positive case): synthetic test record ----
    # Built inline so the self-test is fully portable and runs identically on
    # every machine, with no dependency on any external file. The record
    # exercises the most common code path (HIGH-certainty, primary-gold match,
    # sub-technique with named parent, LOO + BM25 evidence available).
    synthetic_record = {
        'query': 'The malware uses PowerShell to download a script.',
        'query_norm': 'the malware uses powershell to download a script.',
        'actor': 'apt29',
        'prediction': {
            'id': 'T1059.001', 'attack_id': 'T1059.001',
            'score': 5.0, 'correct': True,
        },
        'gold': {'id': 'T1059.001', 'attack_id': 'T1059.001'},
        'confidence': 'HIGH',
        'margin': 3.0,
        'token_importance': [
            {'position': 0, 'token': 'malware',
             'raw_importance': 2.0, 'normalized': 1.0, 'impact': 'HIGH'},
            {'position': 1, 'token': 'PowerShell',
             'raw_importance': 1.5, 'normalized': 0.75, 'impact': 'HIGH'},
            {'position': 2, 'token': 'download',
             'raw_importance': 1.0, 'normalized': 0.5, 'impact': 'MEDIUM'},
        ],
        'bm25_overlap': ['powershell', 'script'],
        'hierarchy': {
            'type': 'sub-technique',
            'parent': 'T1059',
            'parent_name': None,
            'children': [],
            'siblings': [],
        },
        'alternatives': [
            {'rank': 1, 'id': 'T1059.001', 'score': 5.0, 'is_gold': True,
             'text_preview': 'T1059.001 \u2014 PowerShell: '
                             'Adversaries may abuse PowerShell commands...'},
            {'rank': 2, 'id': 'T1059', 'score': 2.0, 'is_gold': False,
             'text_preview': 'T1059 \u2014 Command and Scripting Interpreter: '
                             'Adversaries may abuse...'},
            {'rank': 3, 'id': 'T1105', 'score': 1.0, 'is_gold': False,
             'text_preview': 'T1105 \u2014 Ingress Tool Transfer: '
                             'Adversaries may transfer tools...'},
        ],
    }

    # Determinism: derive_query_state + compose_explanation are pure functions,
    # so two calls on the same input must produce bit-identical output.
    state_a = derive_query_state(synthetic_record)
    state_b = derive_query_state(synthetic_record)
    prose_a = compose_explanation(state_a)
    prose_b = compose_explanation(state_b)
    check("determinism: same input -> bit-identical output", prose_a == prose_b)

    # Validator (positive case): a well-formed paragraph passes validation.
    errs = validate_explanation(prose_a, state_a)
    check("validator accepts a well-formed explanation",
          errs == [], [], errs)

    # ---- Validator: catches forbidden jargon ----
    state = {
        'predicted_id': 'T1059.001',
        'actor_display': 'APT29',
        'confidence_tier_translated': 'high',
    }
    bad_prose = ("In a report attributed to APT29, the logit value for T1059.001 with high "
                 "certainty was annotated. " * 6)
    errs = validate_explanation(bad_prose, state, min_words=20)
    check("validator catches forbidden 'logit'",
          any('logit' in e for e in errs))

    return n_total, len(failures), failures


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate analyst-facing NL explanations from explanations.json"
    )
    parser.add_argument(
        '--input', default='explanations.json',
        help="Path to input explanations.json (default: ./explanations.json)"
    )
    parser.add_argument(
        '--output-dir', default='.',
        help="Output directory (default: current directory)"
    )
    parser.add_argument(
        '--self-test', action='store_true',
        help="Run internal sanity-check suite and exit"
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help="Print one-line summary for each query as it's processed"
    )
    args = parser.parse_args()

    if args.self_test:
        n_total, n_fail, failures = run_self_tests()
        n_pass = n_total - n_fail
        print(f"Self-test: {n_pass}/{n_total} PASS, {n_fail} FAIL")
        for msg in failures:
            print(f"  {msg}")
        sys.exit(0 if n_fail == 0 else 1)

    # Resolve input path: if not found at the given location, try project_files conventions.
    input_path = args.input
    if not os.path.exists(input_path):
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {input_path} ...")
    data = load_explanations(input_path)
    print(f"  Loaded {len(data)} query records")

    # Generate explanations
    records = []
    all_errors = []
    print(f"Generating explanations for {len(data)} queries ...")
    for i, record in enumerate(data):
        state = derive_query_state(record)
        prose = compose_explanation(state)
        errors = validate_explanation(prose, state)
        if errors:
            all_errors.append({
                'index': i,
                'query': state['query'][:60],
                'errors': errors,
            })
        out_rec = build_output_record(state, prose)
        records.append(out_rec)
        if args.verbose:
            status = 'OK' if not errors else 'WARN'
            print(f"  [{status}] {i:3d} {state['predicted_id']:12s} "
                  f"({state['matched_via']:18s}) {state['query'][:50]!r}")

    # Report validation results
    if all_errors:
        print(f"\nValidation issues found in {len(all_errors)} of {len(records)} records:")
        for e in all_errors[:10]:
            print(f"  index {e['index']}: {e['query']!r}")
            for err in e['errors']:
                print(f"     - {err}")
        if len(all_errors) > 10:
            print(f"  ... and {len(all_errors) - 10} more")
    else:
        print(f"\nAll {len(records)} explanations passed validation.")

    # Write nl_explanations.json
    explanations_path = os.path.join(output_dir, 'nl_explanations.json')
    write_json(explanations_path, records)
    print(f"  Wrote {explanations_path}")

    # Write template_versions.json
    templates_path = os.path.join(output_dir, 'template_versions.json')
    write_json(templates_path, build_template_versions_artifact())
    print(f"  Wrote {templates_path}")

    # Write provenance_nl.json
    provenance_path = os.path.join(output_dir, 'provenance_nl.json')
    write_json(provenance_path, build_provenance(input_path, len(records), TEMPLATE_VERSION))
    print(f"  Wrote {provenance_path}")

    # Curate and write nl_examples.md
    curated = curate_paper_examples(records)
    examples_md = render_examples_markdown(curated)
    examples_path = os.path.join(output_dir, 'nl_examples.md')
    write_text(examples_path, examples_md)
    print(f"  Wrote {examples_path} ({len(curated)} curated examples)")

    # Summary statistics for the README
    from collections import Counter
    print("\nDataset statistics:")
    mv = Counter(r['matched_via'] for r in records)
    print(f"  matched_via distribution: {dict(mv)}")
    ct = Counter(r['confidence_tier'] for r in records)
    print(f"  confidence_tier distribution: {dict(ct)}")
    ht = Counter(r['hierarchy_type'] for r in records)
    print(f"  hierarchy_type distribution: {dict(ht)}")
    n_multi = sum(1 for r in records if r['is_multi_gold'])
    print(f"  multi-gold queries: {n_multi}/{len(records)}")
    n_with_bm25 = sum(1 for r in records if r['has_bm25_overlap'])
    print(f"  queries with BM25 overlap: {n_with_bm25}/{len(records)}")
    avg_words = sum(r['n_words'] for r in records) / len(records)
    print(f"  average words per explanation: {avg_words:.1f}")
    avg_sent = sum(r['n_sentences'] for r in records) / len(records)
    print(f"  average sentences per explanation: {avg_sent:.2f}")

    sys.exit(0 if not all_errors else 1)


if __name__ == '__main__':
    main()
