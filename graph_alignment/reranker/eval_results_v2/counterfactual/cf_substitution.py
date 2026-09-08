"""
cf_substitution.py — Counterfactual probe substitute infrastructure for Step D.

This module provides the foundation pieces needed by counterfactual_probe.py:

    1. Surface-form classifier that buckets every actor name (original or
       substitute) by word-count, casing pattern, and digit pattern. Used
       for surface-form-matched substitute selection.
    2. Substitute pool builders for the four pools the user confirmed:
         - Pool A: actor_primary tokens excluding the 7 test actors and
           the 11 curated collision tokens (128 substitutes total).
         - Pool B: curated proper-noun placebos (countries / cities /
           regions) hand-organised into surface-form buckets.
         - Pool C: generic-phrase placebos ("the threat actor",
           "the group", "the adversary").
         - Pool D: malware / software placebos (well-known ATT&CK
           software primaries) per the user's confirmation.
    3. Technique-prior overlap precomputation. For every actor in Pool A
       we derive the set of techniques the actor "uses" from the v14 STIX
       bundle's `relationship` objects, then compute pairwise Jaccard
       similarity to every test-actor's gold technique distribution.
    4. Span identifier that finds the offset of every actor-related token
       in each Stratum A query.
    5. Surface-form-aware substitute selection with progressive fallback
       (per the user's locked-in design from May 2026 conversation).

Design references throughout:
    - Document 12 (Sir-prefix methodological response): substitute pool
      schema, four-placebo design, casing preservation rules.
    - ChatGPT research report: technique-prior overlap as a stratification
      variable rather than hard exclusion.
    - Perplexity Comprehensive Synthesis: surface-form bucket matching
      and frequency-match best practice.
    - Gemini Counterfactual Probing Study Design: candidate-side leak
      audit and explicit confound documentation.
    - Claude operational answers report: progressive fallback policy
      for CamelCase surface forms (CozyDuke, SandWorm).
    - The user's empirical confirmations (May 2026): keep substitute in
      title-case for CamelCase originals (Option A); proper-noun placebo
      as primary D-in-D comparator; report all four placebos separately.

Determinism:
    - All sampling uses np.random.default_rng(seed) with seed surfaced
      as a function argument. The default seed in this module is the
      project-wide SUBSTITUTE_SAMPLING_SEED = 42.
    - Surface-form classification and span identification are pure
      functions with no RNG dependence.

Author: Shane Waldrop, ACSAC 2026 paper, Step D (May 2026).
"""

import json
import re
import os
import sys
from collections import defaultdict, Counter
import numpy as np

SUBSTITUTE_SAMPLING_SEED = 42

# ============================================================================
# Curated proper-noun placebo pool (Pool B)
# ============================================================================
# Hand-curated to match the surface-form buckets that appear in the 21
# Stratum A queries. Names are deliberately neutral, not associated with
# any nation-state cyber operations, and not appearing in the v14 STIX
# bundle as an actor or alias name. Bucket coverage:
#   - 1word_upper_digit (matches APT29 / FIN6 / FIN7 / APT34 surface)
#   - 1word_upper_nodigit (matches COZY surface, record #21)
#   - 1word_title (matches Carbanak / Sandworm / Anunak / Dukes /
#     Cozyduke surface, also CamelCase fallback per user Option A)
#   - 2word_title (matches Wizard Spider surface, but no Stratum A
#     queries use this surface form; included for Stratum B insertion
#     diversity).
#
# Empirical neutrality check (run during pool build): each placebo name
# is verified absent from both the actor_primary and actor_alias token
# sets in vocabulary_v14.json before inclusion.
PROPER_NOUN_POOL_RAW = {
    '1word_upper_digit': [
        # Pseudo-codes matching FIN# / APT## / OilRig-style alphanumeric.
        # We synthesise these because real-world non-cyber proper nouns
        # rarely combine all-caps + a digit in a single token. They are
        # phonologically pronounceable but semantically empty — exactly
        # the property the proper-noun placebo aims to test.
        'ZON12', 'KAR4', 'MIR8', 'VEN21', 'NEX6',
        'OBR15', 'TYL3', 'WAS9', 'XAR7', 'YUR11',
        'CIR23', 'DAR16', 'MOR2', 'NOL5', 'PIR18',
    ],
    '1word_upper_nodigit': [
        # All-caps single words, country / city codes that fit the
        # all-caps no-digit pattern. Aviation airport codes work well
        # because they are real, neutral, and clearly non-cyber.
        'NORDIC', 'PACIFIC', 'ATLANTIC', 'BALTIC',
        'IBERIAN', 'AEGEAN', 'ARCTIC', 'CARIBBEAN',
    ],
    '1word_title': [
        # Title-case 1-word neutral proper nouns: cities and regions
        # that have no cyber-actor associations.
        'Helsinki', 'Madagascar', 'Cordoba', 'Budapest',
        'Reykjavik', 'Marrakech', 'Patagonia', 'Yokohama',
        'Antwerp', 'Tasmania', 'Valencia', 'Bavaria',
        'Lapland', 'Andalusia', 'Catalonia', 'Provence',
    ],
    '2word_title': [
        # Two-word title-case place names matching Wizard Spider
        # surface form, used primarily for Stratum B insertion
        # diversity since wizardspider has zero Stratum A queries.
        'Buenos Aires', 'Hong Kong', 'Cape Town',
        'Costa Rica', 'Sierra Madre', 'Costa Brava',
        'Black Forest', 'Canary Islands',
    ],
}


# ============================================================================
# Curated malware / software placebo pool (Pool D, per user's confirmation)
# ============================================================================
# Well-known ATT&CK software primaries. These ARE real entities in the
# v14 STIX bundle but they are software, not actors — providing a
# "domain-matched but type-mismatched" placebo. This tests whether the
# model is sensitive to "any STIX entity at this slot" vs specifically
# to actor-namespace tokens.
#
# Design note: we prefer software primaries that are tools / commodity
# malware (broad provenance) over targeted-attack-specific implants,
# minimising prior overlap with any single test actor. We also bucket
# by surface form so the same casing-match rules apply.
MALWARE_POOL_RAW = {
    '1word_upper_digit': [
        # Software primaries with all-caps + digits in name. We
        # synthesise additional entries to ensure 5+ substitutes are
        # always available for FIN6/FIN7/APT29/APT34-style originals
        # without falling back to mixed-bucket pool members.
        'MIMI64', 'POW32', 'NJR16', 'GH0ST',
        'NETSH7', 'PSEXEC2', 'WMIC4', 'CMD32',
        'NMAP9', 'TOR1', 'IPCONFIG3', 'WGET8',
    ],
    '1word_upper_nodigit': [
        'MIMIKATZ', 'POWERSHELL', 'BLOODHOUND',
        'METASPLOIT', 'EMPIRE', 'CRACKMAPEXEC',
        'PUPY', 'SLIVER',
    ],
    '1word_title': [
        'Mimikatz', 'PowerShell', 'BloodHound',
        'Empire', 'Sliver', 'Pupy',
        'Meterpreter', 'Rubeus', 'Bitsadmin',
        'Ngrok', 'Plink', 'Psexec',
        'Procdump', 'Snifferpro', 'Lazagne',
    ],
    '2word_title': [
        'Cobalt Strike', 'Brute Ratel', 'Havoc Demon',
        'Sliver Implant',
    ],
}


# ============================================================================
# Generic-phrase placebo pool (Pool C)
# ============================================================================
# Small fixed pool. Selection is deterministic based on grammatical
# context: sentence-initial subject vs sentence-medial reference.
GENERIC_PHRASE_POOL = {
    'sentence_initial': ['The threat actor', 'The group', 'The adversary'],
    'sentence_medial':  ['the threat actor', 'the group', 'the adversary'],
}


# ============================================================================
# The 7 test actors and their canonical names
# ============================================================================
# This map is intentionally hand-coded and verified. The "canonical_name"
# entry is the surface form found in the v14 STIX bundle (used when we
# need to insert the actor's own name into Stratum B queries via Option A).
# The "stix_g_id" is the external_id from the intrusion-set object.
TEST_ACTOR_CANONICAL = {
    'apt29':       {'canonical_name': 'APT29',         'stix_g_id': 'G0016'},
    'carbanak':    {'canonical_name': 'Carbanak',      'stix_g_id': 'G0008'},
    'fin6':        {'canonical_name': 'FIN6',          'stix_g_id': 'G0037'},
    'fin7':        {'canonical_name': 'FIN7',          'stix_g_id': 'G0046'},
    'oilrig':      {'canonical_name': 'OilRig',        'stix_g_id': 'G0049'},
    'sandworm':    {'canonical_name': 'Sandworm Team', 'stix_g_id': 'G0034'},
    'wizardspider':{'canonical_name': 'Wizard Spider', 'stix_g_id': 'G0102'},
}


# ============================================================================
# Surface-form classifier
# ============================================================================
def classify_surface_form(name):
    """
    Classify a proper-noun surface form into one of our buckets.
    
    The classifier is a pure function. It takes the raw surface form
    (e.g., "FIN7", "Carbanak", "CozyDuke", "Wizard Spider") and returns
    a tuple (bucket, attributes_dict).
    
    Buckets: '1word_upper_digit', '1word_upper_nodigit', '1word_title',
             '2word_title', '2word_other', 'mixed_case_special',
             '3plus_words', 'other'
    
    The 'mixed_case_special' bucket captures CamelCase forms like
    "CozyDuke" and "SandWorm" that have no exact-bucket match in the
    actor_primary substitute pool. Per the user's locked-in Option A,
    these get a progressive fallback to '1word_title' with the
    substitute kept in title-case (no re-casting).
    
    Verified test cases (run as __main__ self-test below):
        classify_surface_form('FIN7')         -> '1word_upper_digit'
        classify_surface_form('Carbanak')     -> '1word_title'
        classify_surface_form('CozyDuke')     -> 'mixed_case_special'
        classify_surface_form('SandWorm')     -> 'mixed_case_special'
        classify_surface_form('Wizard Spider')-> '2word_title'
        classify_surface_form('COZY')         -> '1word_upper_nodigit'
        classify_surface_form('Anunak')       -> '1word_title'
    """
    if not name:
        return ('other', {})
    
    words = name.split()
    n_words = len(words)
    has_digit = any(c.isdigit() for c in name)
    is_upper = name.isupper()
    is_title = name.istitle()
    is_lower = name.islower()
    
    # Detect CamelCase: has internal capitals beyond the first character,
    # is not fully upper, is not title-case in the simple sense.
    # "CozyDuke" -> True; "Carbanak" -> False; "APT29" -> False (all upper).
    has_internal_caps = (
        n_words == 1
        and not is_upper
        and not is_lower
        and not is_title
        and any(c.isupper() for c in name[1:])
    )
    
    attrs = {
        'n_words': n_words,
        'has_digit': has_digit,
        'is_upper': is_upper,
        'is_title': is_title,
        'is_lower': is_lower,
        'has_internal_caps': has_internal_caps,
        'len_chars': len(name),
    }
    
    if n_words == 1:
        if has_internal_caps:
            return ('mixed_case_special', attrs)
        if is_upper and has_digit:
            return ('1word_upper_digit', attrs)
        if is_upper and not has_digit:
            return ('1word_upper_nodigit', attrs)
        if is_title:
            return ('1word_title', attrs)
        return ('other', attrs)
    elif n_words == 2:
        if is_title:
            return ('2word_title', attrs)
        return ('2word_other', attrs)
    elif n_words >= 3:
        return ('3plus_words', attrs)
    return ('other', attrs)


# ============================================================================
# Casing preservation: re-cast a substitute to mirror the original's
# surface-form pattern.
# ============================================================================
def recast_to_match(substitute_canonical, original_surface):
    """
    Adjust the casing of a substitute to mirror the original's surface
    form, per the user's Option A confirmation.
    
    User's Option A (May 2026 confirmation): "for camelcase lets do
    option a; keeps the substitute in title-case (substitute 'Andariel'
    stays 'Andariel' even when replacing 'CozyDuke')."
    
    Therefore the rule is:
        - If original is all-upper (FIN7, APT29, COZY): upper-case
          substitute. e.g. 'apt41' -> 'APT41'.
        - If original is title-case (Carbanak, Sandworm, Andariel):
          title-case substitute.
        - If original is CamelCase (CozyDuke, SandWorm): substitute
          STAYS title-case (no re-casting per user Option A). The
          surface-form mismatch is recorded as a covariate.
        - If original is lower-case (rare): lower-case substitute.
    
    Verified test cases:
        recast_to_match('Andariel', 'FIN7')     -> 'ANDARIEL'
        recast_to_match('apt41', 'FIN7')        -> 'APT41'
        recast_to_match('Andariel', 'Carbanak') -> 'Andariel'
        recast_to_match('andariel', 'Carbanak') -> 'Andariel'
        recast_to_match('Andariel', 'CozyDuke') -> 'Andariel' (Option A)
        recast_to_match('Andariel', 'SandWorm') -> 'Andariel' (Option A)
        recast_to_match('APT41',    'COZY')     -> 'APT41' (already upper)
    """
    if not substitute_canonical or not original_surface:
        return substitute_canonical
    
    orig_bucket, orig_attrs = classify_surface_form(original_surface)
    
    if orig_attrs.get('has_internal_caps'):
        # User Option A: keep substitute in its natural title-case form
        # rather than awkwardly re-casing to mirror CamelCase.
        return substitute_canonical.title()
    if orig_attrs.get('is_upper'):
        return substitute_canonical.upper()
    if orig_attrs.get('is_title'):
        return substitute_canonical.title()
    if orig_attrs.get('is_lower'):
        return substitute_canonical.lower()
    # Default: leave substitute as canonically formatted.
    return substitute_canonical


# ============================================================================
# Build the actor_primary substitute pool from vocabulary_v14.json
# ============================================================================
def build_pool_actor_primary(vocab_path, exclude_test_actor_tokens, exclude_collision_tokens):
    """
    Pool A: 128 actor_primary tokens minus the 6 test-actor tokens that
    appear in the vocabulary, minus 'wizard' (Wizard Spider's primary
    token), minus the 11 curated collision tokens.
    
    Each pool entry has:
        norm:           normalised single-token form (lower-case, e.g. 'apt41')
        primary:        canonical surface form from STIX (e.g. 'APT41')
        bucket:         surface-form bucket
        attrs:          {n_words, has_digit, is_upper, ...}
        stix_g_id:      external_id from intrusion-set object
        stix_id:        full STIX UUID
    
    Returns: list of dicts, one per substitute candidate, sorted by norm
    for deterministic ordering.
    """
    with open(vocab_path) as f:
        vocab = json.load(f)
    
    pool = []
    for tok, info in vocab['tokens'].items():
        if info['category'] != 'actor_primary':
            continue
        if tok in exclude_test_actor_tokens:
            continue
        if tok in exclude_collision_tokens:
            continue
        # Get the canonical surface form from provenance
        objects = info.get('provenance', {}).get('objects', [])
        if not objects:
            continue
        primary = objects[0].get('primary_name', tok)
        bucket, attrs = classify_surface_form(primary)
        pool.append({
            'norm': tok,
            'primary': primary,
            'bucket': bucket,
            'attrs': attrs,
            'stix_g_id': objects[0].get('external_id', ''),
            'stix_id': objects[0].get('stix_id', ''),
        })
    
    # Deterministic ordering for reproducibility
    pool.sort(key=lambda r: r['norm'])
    return pool


# ============================================================================
# Build proper-noun and malware pools with bucket-aware structure
# ============================================================================
def build_pool_proper_noun(vocab_path):
    """
    Pool B: curated proper-noun placebos. Verifies neutrality: each
    placebo is checked against vocabulary_v14.json's actor-token sets
    and rejected if it appears as actor_primary or actor_alias.
    
    Returns: list of dicts mirroring the pool A schema.
    """
    with open(vocab_path) as f:
        vocab = json.load(f)
    
    actor_tokens = {tok for tok, info in vocab['tokens'].items()
                    if info['category'] in {'actor_primary', 'actor_alias'}}
    
    pool = []
    for bucket, names in PROPER_NOUN_POOL_RAW.items():
        seen = set()  # de-dupe within bucket
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            tok_norm = name.lower().split()[0] if name else ''
            if tok_norm in actor_tokens:
                # Skip if accidentally collides with a real actor token
                continue
            classified_bucket, attrs = classify_surface_form(name)
            pool.append({
                'norm': name.lower(),
                'primary': name,
                'bucket': classified_bucket,
                'attrs': attrs,
                'stix_g_id': '',
                'stix_id': '',
                'pool_id': 'proper_noun',
            })
    return pool


def build_pool_malware(vocab_path):
    """
    Pool D: curated malware-name placebos per user's confirmation.
    Same structure as Pool B.
    """
    pool = []
    for bucket, names in MALWARE_POOL_RAW.items():
        seen = set()
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            classified_bucket, attrs = classify_surface_form(name)
            pool.append({
                'norm': name.lower(),
                'primary': name,
                'bucket': classified_bucket,
                'attrs': attrs,
                'stix_g_id': '',
                'stix_id': '',
                'pool_id': 'malware',
            })
    return pool


# ============================================================================
# Technique-prior overlap precomputation
# ============================================================================
def build_actor_technique_priors(stix_bundle_path):
    """
    For every intrusion-set in the v14 STIX bundle, derive the set of
    techniques the actor "uses" by following relationship objects of
    type 'uses' between intrusion-set and attack-pattern.
    
    Returns: dict mapping g_id (e.g. 'G0016') to set of attack-pattern
    external_ids (e.g. {'T1059', 'T1566.001', ...}).
    
    The technique sets enable per-pair Jaccard similarity computation
    used for technique-prior-overlap-aware substitute selection.
    
    Verified empirically: APT29 (G0016) should have ~50+ techniques in
    v14; OilRig (G0049) ~30+; FIN6 (G0037) ~15+.
    """
    with open(stix_bundle_path) as f:
        bundle = json.load(f)
    
    # Build reverse maps from STIX UUID to (g_id) and from STIX UUID to
    # technique external_id.
    intrusion_sets = {}  # stix_id -> g_id
    attack_patterns = {}  # stix_id -> technique_external_id
    
    for obj in bundle['objects']:
        if obj.get('revoked') or obj.get('x_mitre_deprecated'):
            continue
        ot = obj.get('type', '')
        if ot == 'intrusion-set':
            for ref in obj.get('external_references', []):
                if ref.get('source_name') == 'mitre-attack':
                    g_id = ref.get('external_id', '')
                    if g_id.startswith('G'):
                        intrusion_sets[obj['id']] = g_id
                        break
        elif ot == 'attack-pattern':
            for ref in obj.get('external_references', []):
                if ref.get('source_name') == 'mitre-attack':
                    t_id = ref.get('external_id', '')
                    if t_id.startswith('T'):
                        attack_patterns[obj['id']] = t_id
                        break
    
    # Walk relationship objects of type 'uses' from intrusion-set to attack-pattern
    actor_techniques = defaultdict(set)
    for obj in bundle['objects']:
        if obj.get('type') != 'relationship':
            continue
        if obj.get('revoked') or obj.get('x_mitre_deprecated'):
            continue
        if obj.get('relationship_type') != 'uses':
            continue
        src = obj.get('source_ref', '')
        tgt = obj.get('target_ref', '')
        if src in intrusion_sets and tgt in attack_patterns:
            g_id = intrusion_sets[src]
            t_id = attack_patterns[tgt]
            actor_techniques[g_id].add(t_id)
    
    # Convert sets to sorted lists for serialisation
    return {g_id: sorted(t_set) for g_id, t_set in actor_techniques.items()}


def jaccard_similarity(set_a, set_b):
    """Standard Jaccard over set intersection / set union."""
    a = set(set_a)
    b = set(set_b)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


# ============================================================================
# Span identification: find actor-token offsets in raw queries
# ============================================================================
def find_actor_spans(query_raw, actor_token_set):
    """
    For a raw query and a set of normalized actor-related tokens, return
    a list of spans where any of the tokens appear (case-insensitive).
    
    Each span is a dict with:
        start:      0-indexed start offset in query_raw
        end:        0-indexed end offset (exclusive)
        surface:    surface-form text in query_raw[start:end]
        normalized: surface.lower()
    
    Word-boundary aware via \\b in the regex. Skips short alias tokens
    (length < 3) to avoid spurious matches.
    
    Verified test cases:
        find_actor_spans("This would allow FIN6 to escalate", {'fin6'})
            -> [{'start': 17, 'end': 21, 'surface': 'FIN6', ...}]
        find_actor_spans("Cozyduke is a malware", {'cozyduke'})
            -> [{'start': 0, 'end': 8, 'surface': 'Cozyduke', ...}]
        find_actor_spans("APT29 has been observed", {'apt29'})
            -> [{'start': 0, 'end': 5, 'surface': 'APT29', ...}]
    """
    if not actor_token_set:
        return []
    safe_tokens = [t for t in actor_token_set if len(t) >= 3]
    if not safe_tokens:
        return []
    pattern = r'\b(' + '|'.join(re.escape(t) for t in sorted(safe_tokens, key=len, reverse=True)) + r')\b'
    spans = []
    for m in re.finditer(pattern, query_raw, flags=re.IGNORECASE):
        spans.append({
            'start': m.start(),
            'end': m.end(),
            'surface': m.group(0),
            'normalized': m.group(0).lower(),
        })
    return spans


# ============================================================================
# Substitute selection algorithm (per the user's locked-in design)
# ============================================================================
def select_substitutes(original_surface, original_actor_g_id, gold_technique_set,
                       pool, actor_techniques, n_substitutes=5,
                       exclude_norms=None, seed=SUBSTITUTE_SAMPLING_SEED,
                       require_bucket_match=True):
    """
    Select N substitutes for the original actor token, applying the
    surface-form bucket match policy with progressive fallback (per
    user Option A) and technique-prior-overlap-aware weighted sampling.
    
    Algorithm:
        1. Classify original_surface into a surface-form bucket.
        2. Filter `pool` to candidates with the same bucket. If empty
           and original is mixed_case_special (CamelCase like CozyDuke),
           fall back to '1word_title' bucket (per user Option A).
           If still empty, fall back to all pool members regardless of
           bucket.
        3. Exclude any substitutes in `exclude_norms` (e.g. the test
           actor's own family aliases).
        4. Score each remaining candidate by 1 - jaccard_similarity
           between the candidate's technique set and the query's gold
           techniques. Lower overlap -> higher score, so we
           preferentially sample substitutes whose technique-priors
           don't overlap with the gold (more diagnostic).
        5. Sample n_substitutes deterministically using
           np.random.default_rng(seed), weighted by score with no
           replacement.
        6. Re-cast each substitute's surface form to mirror the
           original's casing pattern (per recast_to_match()).
    
    Returns: list of n_substitutes dicts, each with:
        {
          'norm':                normalized substitute token
          'primary':             canonical substitute name from pool
          'recast':              cased substitute matching original
          'bucket':              substitute's own bucket
          'fallback_level':      0=exact, 1=fallback to 1word_title,
                                 2=any-bucket fallback
          'tech_jaccard':        Jaccard similarity to gold techniques
          'subword_token_delta': PLACEHOLDER -1 (filled by probe pipeline
                                 when tokenizer is available)
        }
    """
    if exclude_norms is None:
        exclude_norms = set()
    
    orig_bucket, _ = classify_surface_form(original_surface)
    
    # Per the user's Option A confirmation: CamelCase originals
    # (CozyDuke, SandWorm, OilRig) draw from 1word_title with the
    # substitute kept in title-case. We treat mixed_case_special as a
    # routing alias to 1word_title rather than searching for CamelCase
    # substitutes. Other CamelCase substitutes do exist in the pool
    # (SideCopy, LazyScripter, MuddyWater, etc.) but the user explicitly
    # chose to use natural title-case substitutes instead.
    target_bucket = '1word_title' if orig_bucket == 'mixed_case_special' else orig_bucket
    
    # Phase 1: try target bucket match
    candidates = [p for p in pool
                  if p['bucket'] == target_bucket
                  and p['norm'] not in exclude_norms]
    fallback_level = 1 if orig_bucket == 'mixed_case_special' else 0
    
    # Phase 2: ultimate fallback to any pool member
    if len(candidates) < n_substitutes:
        more = [p for p in pool
                if p['norm'] not in exclude_norms
                and p['norm'] not in {c['norm'] for c in candidates}]
        candidates = candidates + more
        if fallback_level < 2:
            fallback_level = 2
    
    if not candidates:
        return []
    
    # Score by 1 - Jaccard with gold techniques
    if require_bucket_match and fallback_level == 0:
        scoring_set = candidates
    else:
        scoring_set = candidates
    
    scores = []
    for c in scoring_set:
        cand_g_id = c.get('stix_g_id', '')
        cand_tech_set = set(actor_techniques.get(cand_g_id, []))
        gold_set = set(gold_technique_set)
        # Filter gold set to T-prefixed entries (techniques only, drop
        # tactics like TA0002 and software like S0030 that won't be in
        # actor_techniques)
        gold_t_only = {g for g in gold_set if g.startswith('T') and not g.startswith('TA')}
        if not gold_t_only or not cand_tech_set:
            # If either side empty, neutral score
            score = 0.5
        else:
            jacc = jaccard_similarity(cand_tech_set, gold_t_only)
            # Prefer LOWER jaccard (less prior overlap)
            score = 1.0 - jacc
        scores.append(score)
    
    # Weighted-by-score deterministic sampling without replacement
    rng = np.random.default_rng(seed)
    scores_arr = np.array(scores)
    # Normalise to probabilities; handle degenerate all-zero case
    if scores_arr.sum() <= 0:
        probs = np.ones(len(scores_arr)) / len(scores_arr)
    else:
        probs = scores_arr / scores_arr.sum()
    
    n_pick = min(n_substitutes, len(scoring_set))
    picked_idx = rng.choice(len(scoring_set), size=n_pick, replace=False, p=probs)
    
    selected = []
    for idx in picked_idx:
        cand = scoring_set[int(idx)]
        cand_g_id = cand.get('stix_g_id', '')
        cand_tech_set = set(actor_techniques.get(cand_g_id, []))
        gold_t_only = {g for g in gold_technique_set
                       if g.startswith('T') and not g.startswith('TA')}
        jacc = (jaccard_similarity(cand_tech_set, gold_t_only)
                if (gold_t_only and cand_tech_set) else float('nan'))
        recast_form = recast_to_match(cand['primary'], original_surface)
        selected.append({
            'norm': cand['norm'],
            'primary': cand['primary'],
            'recast': recast_form,
            'bucket': cand['bucket'],
            'fallback_level': fallback_level,
            'tech_jaccard': float(jacc),
            'subword_token_delta': -1,  # filled by tokenizer in probe pipeline
        })
    return selected


# ============================================================================
# Generic-phrase placebo selection
# ============================================================================
def select_generic_phrase(original_span_start, n_substitutes=3,
                           preceding_text=None):
    """
    Pool C: pick a generic phrase based on whether the original span
    is sentence-initial (start == 0) or sentence-medial.
    
    Article-collision handling: if the span is preceded by "the " or
    "The " (article followed by space), we use the article-less form
    of each phrase to avoid duplicate articles like "The the adversary".
    
    Empirical: 1 of 21 Stratum A queries triggers this (record #2,
    "The Dukes are known...") where span 'Dukes' is at offset 4 preceded
    by "The ". The fix removes "the " from the substitute so the
    sentence reads "The threat actor are known..." rather than "The
    the threat actor are known...".
    
    Args:
        original_span_start: 0-indexed start offset of the actor span.
        n_substitutes: cap on number of phrases (Pool C has 3 each).
        preceding_text: optional 4-char window before the span; if it
            ends with 'the ' or 'The ', article-collision avoidance
            kicks in.
    
    Returns: list of substitute dicts (up to n_substitutes; 3 phrases
    per context).
    """
    is_initial = (original_span_start == 0)
    has_preceding_article = (
        preceding_text is not None and
        len(preceding_text) >= 4 and
        preceding_text[-4:].lower() == 'the '
    )
    
    if is_initial:
        phrases = GENERIC_PHRASE_POOL['sentence_initial']
    elif has_preceding_article:
        # Use article-less forms: "threat actor", "group", "adversary"
        phrases = ['threat actor', 'group', 'adversary']
    else:
        phrases = GENERIC_PHRASE_POOL['sentence_medial']
    
    selected = []
    for phrase in phrases[:n_substitutes]:
        selected.append({
            'norm': phrase.lower(),
            'primary': phrase,
            'recast': phrase,  # already context-appropriate
            'bucket': 'generic_phrase',
            'fallback_level': 0,
            'tech_jaccard': float('nan'),
            'subword_token_delta': -1,
        })
    return selected


# ============================================================================
# Soft-deletion substitute (Pool E - deletion as fifth perturbation type)
# ============================================================================
def select_soft_deletion(original_span_start, original_surface):
    """
    Soft deletion: replace the actor span with the mask token '[MASK]'.
    
    This is the C5 condition. Per the consensus across all eleven
    research documents (notably Claude research report citing Heimersheim
    & Nanda 2024 arXiv:2404.15255 and Zhang & Nanda 2023 arXiv:2309.16042),
    the canonical mechanistic-interpretability soft-deletion form is the
    tokenizer's mask token. This:
        - preserves a single subword token in the input (length control)
        - is unambiguously distinct from generic_phrase_placebo (C3),
          which uses natural-language entity-class placeholders ("the
          threat actor", "the group", "the adversary")
        - tests the model's response to information removal rather than
          to entity-class substitution
        - is BERT/MiniLM tokenizer-native: the tokenizer treats '[MASK]'
          as a single special token (subword index 103 in BERT-base
          vocab; verified for MiniLM L6-H384).
    
    Note: we use '[MASK]' as the literal string. The cross-encoder
    tokenizer recognises this and assigns it the mask-token ID rather
    than tokenizing it as bracketed text. This matches the
    activation-patching convention used in Heimersheim & Nanda 2024.
    
    For sentence-initial subjects we capitalise the token boundary
    naturally; the bracket form is the same regardless of position.
    
    Returns: list of exactly 1 substitute (deletion is deterministic).
    """
    # The mask token is the same regardless of sentence position. The
    # [MASK] form is the canonical BERT-family special token.
    replacement = '[MASK]'
    return [{
        'norm': '[mask]',
        'primary': replacement,
        'recast': replacement,
        'bucket': 'soft_deletion',
        'fallback_level': 0,
        'tech_jaccard': float('nan'),
        'subword_token_delta': -1,
    }]


# ============================================================================
# Apply a substitute to a query (perturbation step)
# ============================================================================
def apply_substitute(query_raw, span, substitute_recast):
    """
    Replace query_raw[span['start']:span['end']] with substitute_recast.
    Preserves all surrounding text exactly.
    
    Returns: perturbed query_raw string.
    """
    return query_raw[:span['start']] + substitute_recast + query_raw[span['end']:]


def apply_insertion(query_raw, actor_name, insertion_template='[ACTOR_NAME] was observed '):
    """
    Per user's Option A confirmation: prepend "[ACTOR_NAME] was observed "
    at the start of the query.
    
    Lower-cases the original query's first letter so the result reads
    grammatically (e.g. "stealing files" rather than "Stealing files"
    after the prepend).
    
    The lowercase rule is "Normal Capitalized word detection". We only
    lowercase the first character when the first word looks like a
    plain sentence-initial capitalisation, meaning:
      (a) the first character is uppercase, AND
      (b) the second character is a lowercase letter, AND
      (c) the remainder of the first word contains no further uppercase
          letters (so we preserve brand names with internal capitals
          like PowerShell, JavaScript, MongoDB).
    
    Verified test cases (in _self_test):
      "Stealing files"            -> "stealing files"           (normal)
      "Microsoft Excel"           -> "microsoft Excel"          (normal)
      "VBScripts."                -> "VBScripts."               (acronym preserved)
      "GH0ST"                     -> "GH0ST"                    (digit-acronym preserved)
      "FIN7 incorporated"         -> "FIN7 incorporated"        (all-caps preserved)
      "I went to..."              -> "I went to..."             (single letter preserved)
      "PowerShell was used"       -> "PowerShell was used"      (brand name preserved)
      "JavaScript code injection" -> "JavaScript code injection" (brand name preserved)
    
    Returns: (perturbed_query, insertion_text)
    """
    prefix = insertion_template.replace('[ACTOR_NAME]', actor_name)
    body = query_raw
    if query_raw and query_raw[0].isupper():
        if len(query_raw) >= 2 and query_raw[1].isalpha() and query_raw[1].islower():
            # Now check whether the first word has internal capitals.
            # If it does, it is a brand name like "PowerShell" and we
            # preserve the case. Otherwise it is a normal Capitalized
            # word ("Stealing", "Microsoft") and we lowercase.
            i = 1
            while i < len(query_raw) and (query_raw[i].isalpha() or query_raw[i].isdigit()):
                i += 1
            first_word_tail = query_raw[1:i]
            if not any(c.isupper() for c in first_word_tail):
                body = query_raw[0].lower() + query_raw[1:]
    return (prefix + body), prefix


# ============================================================================
# Self-tests for this module
# ============================================================================
def _self_test():
    """Verification suite that runs when this module is invoked directly."""
    print("=" * 75)
    print("cf_substitution.py self-test")
    print("=" * 75)
    
    # Test 1: surface-form classifier on all test actor canonical names
    print("\nTest 1: classify_surface_form() on test actor canonical names")
    expected = {
        'APT29':         '1word_upper_digit',
        'Carbanak':      '1word_title',
        'FIN6':          '1word_upper_digit',
        'FIN7':          '1word_upper_digit',
        'OilRig':        'mixed_case_special',
        'Sandworm Team': '2word_title',
        'Wizard Spider': '2word_title',
        'CozyDuke':      'mixed_case_special',
        'SandWorm':      'mixed_case_special',
        'Cozyduke':      '1word_title',  # not internal-caps, just title
        'COZY':          '1word_upper_nodigit',
        'Dukes':         '1word_title',
        'Anunak':        '1word_title',
        'APT34':         '1word_upper_digit',
    }
    for name, expected_bucket in expected.items():
        bucket, _ = classify_surface_form(name)
        status = 'PASS' if bucket == expected_bucket else 'FAIL'
        print(f"  [{status}] '{name}' -> {bucket} (expected {expected_bucket})")
        if bucket != expected_bucket:
            raise AssertionError(f"surface-form classifier failed on '{name}'")
    
    # Test 2: recast_to_match per user's Option A
    print("\nTest 2: recast_to_match() per user's Option A")
    recast_cases = [
        ('Andariel', 'FIN7',     'ANDARIEL'),     # all-upper original
        ('apt41',    'FIN7',     'APT41'),
        ('Andariel', 'Carbanak', 'Andariel'),     # title-case original
        ('andariel', 'Carbanak', 'Andariel'),
        ('Andariel', 'CozyDuke', 'Andariel'),     # CamelCase Option A
        ('Andariel', 'SandWorm', 'Andariel'),     # CamelCase Option A
        ('APT41',    'COZY',     'APT41'),
        ('Lazarus',  'APT29',    'LAZARUS'),
    ]
    for sub, orig, expected in recast_cases:
        got = recast_to_match(sub, orig)
        status = 'PASS' if got == expected else 'FAIL'
        print(f"  [{status}] recast({sub!r}, {orig!r}) -> {got!r} (expected {expected!r})")
        if got != expected:
            raise AssertionError(f"recast failed on ({sub!r}, {orig!r})")
    
    # Test 3: find_actor_spans
    print("\nTest 3: find_actor_spans()")
    span_cases = [
        ("This would allow FIN6 to escalate", {'fin6'}, [(17, 21, 'FIN6')]),
        ("FIN6 generally used either",        {'fin6'}, [(0, 4, 'FIN6')]),
        ("CozyDuke is a malware",             {'cozyduke'}, [(0, 8, 'CozyDuke')]),
        ("APT29 has been observed",           {'apt29'}, [(0, 5, 'APT29')]),
        ("hosting CozyDuke",                  {'cozyduke'}, [(8, 16, 'CozyDuke')]),
        ("the malware",                       {'fin6'}, []),
    ]
    for query, tokens, expected_spans in span_cases:
        spans = find_actor_spans(query, tokens)
        actual = [(s['start'], s['end'], s['surface']) for s in spans]
        status = 'PASS' if actual == expected_spans else 'FAIL'
        print(f"  [{status}] find_actor_spans({query[:30]!r}..., {tokens}) -> {actual}")
        if actual != expected_spans:
            raise AssertionError(f"span finder failed on {query!r}")
    
    # Test 4: apply_substitute
    print("\nTest 4: apply_substitute() preserves surrounding text exactly")
    q = "This would allow FIN6 to escalate privileges in the system."
    span = {'start': 17, 'end': 21, 'surface': 'FIN6', 'normalized': 'fin6'}
    perturbed = apply_substitute(q, span, 'APT41')
    expected = "This would allow APT41 to escalate privileges in the system."
    status = 'PASS' if perturbed == expected else 'FAIL'
    print(f"  [{status}] perturbed: {perturbed!r}")
    if perturbed != expected:
        raise AssertionError("apply_substitute failed")
    
    # Test 5: apply_insertion (user's Option A)
    print("\nTest 5: apply_insertion() prepends '[ACTOR_NAME] was observed '")
    q = "Stealing files"
    perturbed, prefix = apply_insertion(q, 'APT29')
    expected = "APT29 was observed stealing files"
    status = 'PASS' if perturbed == expected else 'FAIL'
    print(f"  [{status}] perturbed: {perturbed!r} (prefix: {prefix!r})")
    if perturbed != expected:
        raise AssertionError("apply_insertion failed for normal-case start")
    
    # Test 5b: insertion preserves leading all-caps
    q2 = "FIN7 incorporated and adapted"
    perturbed2, _ = apply_insertion(q2, 'APT29')
    expected2 = "APT29 was observed FIN7 incorporated and adapted"
    status = 'PASS' if perturbed2 == expected2 else 'FAIL'
    print(f"  [{status}] preserves all-caps start: {perturbed2!r}")
    if perturbed2 != expected2:
        raise AssertionError("apply_insertion failed for all-caps start")
    
    # Test 5c: insertion preserves CamelCase / acronym (the VBScripts. bug)
    q3 = "VBScripts."
    perturbed3, _ = apply_insertion(q3, 'OilRig')
    expected3 = "OilRig was observed VBScripts."
    status = 'PASS' if perturbed3 == expected3 else 'FAIL'
    print(f"  [{status}] preserves CamelCase/acronym: {perturbed3!r}")
    if perturbed3 != expected3:
        raise AssertionError("apply_insertion failed for CamelCase/acronym (the VBScripts bug)")
    
    # Test 5d: insertion preserves all-caps token like "GH0ST"
    q4 = "GH0ST malware sample"
    perturbed4, _ = apply_insertion(q4, 'APT29')
    expected4 = "APT29 was observed GH0ST malware sample"
    status = 'PASS' if perturbed4 == expected4 else 'FAIL'
    print(f"  [{status}] preserves digit-acronym start: {perturbed4!r}")
    if perturbed4 != expected4:
        raise AssertionError("apply_insertion failed for digit-acronym start")
    
    # Test 5e: insertion handles single-letter start "I went..." correctly
    q5 = "I went to the conference"
    perturbed5, _ = apply_insertion(q5, 'APT29')
    expected5 = "APT29 was observed I went to the conference"
    status = 'PASS' if perturbed5 == expected5 else 'FAIL'
    print(f"  [{status}] preserves single-letter capital: {perturbed5!r}")
    if perturbed5 != expected5:
        raise AssertionError("apply_insertion failed for single-letter capital")
    
    # Test 5f: insertion handles Microsoft case correctly (not all-caps)
    q6 = "Microsoft Excel macros"
    perturbed6, _ = apply_insertion(q6, 'OilRig')
    expected6 = "OilRig was observed microsoft Excel macros"
    status = 'PASS' if perturbed6 == expected6 else 'FAIL'
    print(f"  [{status}] lowercases first word only: {perturbed6!r}")
    if perturbed6 != expected6:
        raise AssertionError("apply_insertion failed for normal capital + later capitals")
    
    # Test 5g: insertion preserves brand names with internal capitals (PowerShell)
    q7 = "PowerShell was used heavily for execution"
    perturbed7, _ = apply_insertion(q7, 'Carbanak')
    expected7 = "Carbanak was observed PowerShell was used heavily for execution"
    status = 'PASS' if perturbed7 == expected7 else 'FAIL'
    print(f"  [{status}] preserves brand-name CamelCase: {perturbed7!r}")
    if perturbed7 != expected7:
        raise AssertionError("apply_insertion failed for brand-name CamelCase (PowerShell case)")
    
    # Test 5h: insertion preserves JavaScript brand name
    q8 = "JavaScript code injected into the page"
    perturbed8, _ = apply_insertion(q8, 'FIN7')
    expected8 = "FIN7 was observed JavaScript code injected into the page"
    status = 'PASS' if perturbed8 == expected8 else 'FAIL'
    print(f"  [{status}] preserves JavaScript brand name: {perturbed8!r}")
    if perturbed8 != expected8:
        raise AssertionError("apply_insertion failed for JavaScript brand name")
    
    # Test 6: jaccard_similarity
    print("\nTest 6: jaccard_similarity()")
    j_cases = [
        ({'a','b','c'}, {'a','b','c'}, 1.0),
        ({'a','b','c'}, {'a'}, 1.0/3.0),
        (set(), set(), 0.0),
        ({'a','b'}, {'c','d'}, 0.0),
        ({'a','b'}, {'a','c'}, 1.0/3.0),
    ]
    for a, b, exp in j_cases:
        got = jaccard_similarity(a, b)
        status = 'PASS' if abs(got - exp) < 1e-9 else 'FAIL'
        print(f"  [{status}] jaccard({a}, {b}) = {got:.4f} (expected {exp:.4f})")
        if abs(got - exp) >= 1e-9:
            raise AssertionError(f"jaccard failed on ({a}, {b})")
    
    print("\n" + "=" * 75)
    print("All cf_substitution self-tests PASSED")
    print("=" * 75)


if __name__ == '__main__':
    _self_test()
