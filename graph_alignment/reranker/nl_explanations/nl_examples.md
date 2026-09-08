# Curated Natural-Language Explanation Examples

These examples are selected to illustrate the explanation pattern across
the case space: confidence tiers, multi-gold cases, error cases, and the
short-query fallback. Complete explanations for all 146 test queries are
in `nl_explanations.json`.

## Example 1: HIGH confidence, single-gold, primary_gold match

*Why this example:* the easy case: clean single-label match with large score gap.

**Query (verbatim from source CTI report):**

> The Macros contain several anti-VM checks designed to avoid executing in virtualized environments

**Structured signals consumed by the template:**

- Actor (metadata, not model input): `apt29` -> `APT29`
- Predicted: `T1497` (Virtualization/Sandbox Evasion)
- Confidence: `HIGH` (translated to `high`)
- Confidence score: `6.5708`
- Score gap to next: `9.7852`
- Matched via: `primary_gold`
- All gold IDs (n=1): `['T1497']`
- Top LOO evidence tokens: `['checks', 'virtualized', 'avoid']`
- BM25 overlap (post-dedup): `['environments', 'several']`
- Hierarchy: `technique`

**Generated explanation paragraph:**

In a report attributed to APT29, the reranker assigned **T1497 (Virtualization/Sandbox Evasion)** as its top-ranked candidate with high certainty (confidence score 6.57, score gap of 9.79 over the next candidate). The strongest token-level evidence comes from the terms 'checks', 'virtualized', and 'avoid', with additional lexical overlap against the candidate description on 'environments' and 'several'. **T1497** is the single ATT&CK technique annotated for this query. The next-ranked candidate was **T1569.002 (Service Execution)** at confidence score -3.21; in structural terms, T1497 is a top-level ATT&CK technique. This is a high-certainty assignment suitable for downstream automation with light spot-checking.

*(5 sentences, 96 words)*

---

## Example 2: HIGH confidence, multi-gold, alternative_gold match

*Why this example:* the dominant case: model picked one of multiple valid annotations.

**Query (verbatim from source CTI report):**

> Carbanak uses the HTTP protocol

**Structured signals consumed by the template:**

- Actor (metadata, not model input): `carbanak` -> `Carbanak`
- Predicted: `S0030` (Carbanak)
- Confidence: `HIGH` (translated to `high`)
- Confidence score: `6.9805`
- Score gap to next: `9.1541`
- Matched via: `alternative_gold`
- All gold IDs (n=2): `['S0030', 'TA0010']`
- Top LOO evidence tokens: `['Carbanak']`
- BM25 overlap (post-dedup): `[]`
- Hierarchy: `software`

**Generated explanation paragraph:**

In a report attributed to Carbanak, the reranker assigned **S0030 (Carbanak)** as its top-ranked candidate with high certainty (confidence score 6.98, score gap of 9.15 over the next candidate). The strongest token-level evidence is the term 'Carbanak'. **S0030** is one of 2 ATT&CK techniques annotated as a valid mapping for this query. The next-ranked candidate was **TA0008 (Lateral Movement)** at confidence score -2.17; in structural terms, S0030 is an ATT&CK software entry. This is a high-certainty assignment suitable for downstream automation with light spot-checking.

*(5 sentences, 84 words)*

---

## Example 3: MEDIUM confidence, correct

*Why this example:* shows hedged language and brief-review recommendation.

**Query (verbatim from source CTI report):**

> It relies on the Gri"on JScript backdoor

**Structured signals consumed by the template:**

- Actor (metadata, not model input): `fin7` -> `FIN7`
- Predicted: `T1059` (Command and Scripting Interpreter)
- Confidence: `MEDIUM` (translated to `moderate`)
- Confidence score: `4.4318`
- Score gap to next: `0.2971`
- Matched via: `alternative_gold`
- All gold IDs (n=3): `['T1059', 'TA0002', 'S0417']`
- Top LOO evidence tokens: `['JScript', 'backdoor', 'Gri"on']`
- BM25 overlap (post-dedup): `[]`
- Hierarchy: `technique`

**Generated explanation paragraph:**

In a report attributed to FIN7, the reranker assigned **T1059 (Command and Scripting Interpreter)** as its top-ranked candidate with moderate certainty (confidence score 4.43, score gap of 0.30 over the next candidate). The strongest token-level evidence comes from the terms 'JScript', 'backdoor', and 'Gri"on'. **T1059** is one of 3 ATT&CK techniques annotated as a valid mapping for this query; the primary annotation is **TA0002 (Execution)**. In structural terms, T1059 is a top-level ATT&CK technique. This is a moderate-certainty assignment; brief analyst review is recommended before downstream automation.

*(5 sentences, 87 words)*

---

## Example 4: Error case, gold at rank 2 (the typical error pattern)

*Why this example:* shows error framing with the annotated technique nearby.

**Query (verbatim from source CTI report):**

> then invokes a WMI instance in the rootsecuritycenter namespace to identify security products installed on the system

**Structured signals consumed by the template:**

- Actor (metadata, not model input): `apt29` -> `APT29`
- Predicted: `T1518.001` (Security Software Discovery)
- Confidence: `MEDIUM` (translated to `moderate`)
- Confidence score: `5.5316`
- Score gap to next: `0.2110`
- Matched via: `error`
- All gold IDs (n=2): `['T1047', 'TA0002']`
- Top LOO evidence tokens: `['identify', 'security']`
- BM25 overlap (post-dedup): `['installed', 'system']`
- Hierarchy: `sub-technique`, parent `T1518`

**Generated explanation paragraph:**

In a report attributed to APT29, the reranker assigned **T1518.001 (Security Software Discovery)** as its top-ranked candidate with moderate certainty (confidence score 5.53, score gap of 0.21 over the next candidate). The strongest token-level evidence comes from the terms 'identify' and 'security', with additional lexical overlap against the candidate description on 'installed' and 'system'. However, the annotated technique **T1047 (Windows Management Instrumentation)** ranked 2nd in the model's candidate list with confidence score 5.32. In structural terms, T1518.001 is a sub-technique of **T1518**. Given the moderate certainty and the small score gap, analyst review is recommended to reconcile the disagreement with the annotated mapping.

*(5 sentences, 103 words)*

---

## Example 5: Software / short-query, BM25-only evidence

*Why this example:* shows the no-LOO evidence fallback path.

**Query (verbatim from source CTI report):**

> Trickbot,

**Structured signals consumed by the template:**

- Actor (metadata, not model input): `wizardspider` -> `Wizard Spider`
- Predicted: `S0266` (TrickBot)
- Confidence: `HIGH` (translated to `high`)
- Confidence score: `9.4017`
- Score gap to next: `7.3267`
- Matched via: `primary_gold`
- All gold IDs (n=1): `['S0266']`
- Top LOO evidence tokens: `[]`
- BM25 overlap (post-dedup): `['trickbot']`
- Hierarchy: `software`

**Generated explanation paragraph:**

In a report attributed to Wizard Spider, the reranker assigned **S0266 (TrickBot)** as its top-ranked candidate with high certainty (confidence score 9.40, score gap of 7.33 over the next candidate). With a short query, individual token-impact attribution does not apply; the principal evidence is lexical overlap with the candidate description on 'trickbot'. **S0266** is the single ATT&CK technique annotated for this query. The next-ranked candidate was **TA0003 (Persistence)** at confidence score 2.08; in structural terms, S0266 is an ATT&CK software entry. This is a high-certainty assignment suitable for downstream automation with light spot-checking.

*(5 sentences, 93 words)*

---
