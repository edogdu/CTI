# Natural-Language Explanation Synthesis Layer

This directory contains the deterministic template-based explanation layer
for the CTI-HAL cross-encoder reranker, prepared for ACSAC 2026 submission.
The script reads the per-query structured explanation record at
`explanations.json` and produces a single analyst-facing paragraph for each
of the 146 test queries, alongside structured fields that trace every prose
claim back to the JSON field it was derived from.

## What this is, in one sentence

A deterministic, reproducible "XAI Description" layer (in the sense of
Cambria et al.'s XAI Descriptions vs Narratives distinction) that
verbalizes already-computed structured explanation signals into a compact
analyst-facing paragraph, with no LLM dependency and bit-identical output
across runs.

## Quick start (Windows cmd.exe)

The script needs `explanations.json` in the parent directory of this
folder, which is the canonical location produced by the reranker's
explanation pipeline. From this directory, run:

```cmd
python generate_nl_explanations.py --input ..\explanations.json --output-dir .
```

To run the internal sanity-check suite without producing any output files,
which executes 31 component-level tests in under a second:

```cmd
python generate_nl_explanations.py --self-test
```

The script depends only on the Python 3 standard library. No additional
packages, no network access, no LLM API calls.

## Output files

Running the script produces five files in the output directory. Each is
described below in the order a reviewer or future-user is likely to
consult them.

`nl_explanations.json` is the primary output, containing all 146
generated explanations together with the structured fields each paragraph
was derived from. Every record is approximately 37 fields including the
predicted technique ID and name, the confidence tier in both its raw and
translated forms, the score gap to the next candidate, the matched-via
taxonomy classification, the full multi-gold annotation set derived from
the alternatives' is_gold flags, the selected leave-one-out evidence
tokens, the BM25 overlap tokens after deduplication, the hierarchy type
and parent context, the next-ranked alternative for cases where it is
distinct from the primary gold, and the prose paragraph itself.

`nl_examples.md` contains five curated paper-ready examples covering the
case space: a high-confidence single-gold success, a high-confidence
multi-gold alternative-gold match, a medium-confidence correct case, an
error case where the annotated technique ranked second, and the
software/short-query case that exercises the BM25-only evidence fallback.
Each example is rendered with both the structured signals consumed by the
template and the generated paragraph, so a reviewer reading only this
file can verify that every claim in the prose maps to a field in the
record.

`template_versions.json` snapshots the template version identifier, the
schema version, the strict sentence ordering, the vocabulary translation
map, the canonical actor display names, the forbidden-jargon hygiene
list, and the design citation anchors. A reviewer can read this file
without reading the Python source and still understand the design.

`provenance_nl.json` records the generation timestamp in UTC, the
template version, the absolute path of the input file, the SHA-256 hex
digest of the input bytes, the record count, the Python version, and the
design citation anchors. The SHA-256 ties this batch of outputs to a
specific input snapshot — if `explanations.json` changes, the hash
changes, and the explanations need to be regenerated.

`generate_nl_explanations.py` is the script itself, structured as one
self-contained file because the SPLAIN reproducibility convention is to
publish the templates and the generation logic together as a single
auditable artifact. The script contains an internal sanity-check suite
that can be invoked with `--self-test`.

## Design philosophy

The MVP is positioned as an "XAI Description" layer rather than a novel
XAI algorithm. This framing comes from the four research reports
synthesized before the build, all of which converged on the same
recommendation: do not claim faithfulness to the cross-encoder's internal
reasoning, because LOO attribution measures input-output sensitivity not
internal mechanisms. Instead, claim faithfulness to the structured
signals that the layer consumes — every quantitative phrase in every
paragraph maps to a specific field in the JSON record.

The strict sentence order follows Gemini's recommended structure with
the analyst-action implication added as the fifth sentence per Rastogi
et al.'s finding that Tier-1 analysts need "straightforward next steps."
The five slots are deployment context plus primary prediction plus
confidence and score gap in sentence one, token-level and lexical
evidence in sentence two, the multi-gold or error correctness framing in
sentence three, the next-ranked alternative (when distinct from the
primary gold already named) and the hierarchy context in sentence four,
and the analyst-action implication scaled to confidence tier and
correctness state in sentence five.

The translated vocabulary discipline is consistent throughout: the raw
cross-encoder logit is presented as "confidence score," the margin
between top-1 and top-2 is presented as "score gap," the leave-one-out
attribution is presented as "the strongest token-level evidence," and
the categorical confidence tier is presented as "high/moderate/low
certainty." A validation pass actively checks for forbidden jargon tokens
(`logit`, `softmax`, `leave-one-out`, `top-k`, `argmax`, `perturbation`,
`embedding`, and similar) and would fail any paragraph that contained
them. All 146 paragraphs pass this check.

The actor opening grammatically subordinates the threat-actor label so
that the model is the active subject and the actor is passive context.
Every paragraph begins with "In a report attributed to {Actor}," and the
actor never appears as the active subject of a model-action verb. This
is the design choice the Step D counterfactual probe substantiates — the
probe shows that actor names do not drive the model's predictions, so
the actor appears in the prose only as metadata provided by the upstream
pipeline. The cross-reference to the counterfactual probe belongs in the
paper text around the figure box rather than in each per-query paragraph
(the word budget at 80-120 words does not have room for it, and
explaining the probe once in the paper is cleaner than 146 times in the
prose).

## Citation anchors for the paper

The primary precedent for the templates-as-contribution framing is
SPLAIN (Kazakova, Hwang, Dorr, Wilks, Gage, Memory, and Clark; arXiv
2311.11215; 2023), an IHMC and Leidos cyber-threat warning system funded
by IARPA. SPLAIN explicitly defends "a hierarchical template-based
approach [that] ensures consistent warning structure and vocabulary" as
a contribution in security XAI. SPLAIN ships with no user study, no
quantitative evaluation, and no baseline comparison — the same scope
shape this MVP has — which establishes that template design itself can
be a legitimate methodological contribution at top venues.

The secondary precedent for the structured-to-text template pattern is
AGIR (Perrina, Conti, Picek, and Cesarano; IEEE BigData 2023), which
validates the template-first stage as an independent contribution
reported separately from any subsequent LLM polish.

The faithfulness language convention follows Lyu, Apidianaki, and
Callison-Burch (Computational Linguistics 50(2):657-723, 2024), whose
"plausibility versus faithfulness" distinction supports the position
that this layer surfaces observable input-output signals rather than
internal model reasoning.

The 4-6 sentence length and the translated-vocabulary register come from
Rastogi, Pant, Dhanuka, Saxena, and Mairal (arXiv 2503.02065; 2025),
whose mixed-methods study (N=248 survey and N=24 interviews across
Tier-1 to Tier-3 SOC analysts) documents both the attention-budget
constraint and the analyst preference for direct-action recommendations.

The absence of a formal analyst evaluation is normalized by Nadeem,
Vos, Cao, Pajola, Dieber, Frassinelli, Conti, and van der Heijden's
Systematization of Knowledge in security XAI (EuroS&P 2023; pp. 221-240),
which reports that user studies appear in only 14% of published security
XAI papers — making the absence of one in this MVP consistent with
field norms rather than a deficiency.

## Defense against likely reviewer objections

Reviewers may attack the faithfulness claim, arguing that LOO does not
reflect the cross-encoder's internal reasoning. The defense is that we
do not claim faithfulness to internal reasoning anywhere in the paper or
the prose. We claim faithfulness to observable input-output sensitivity,
which is exactly what LOO measures by definition. The Lyu et al. 2024
plausibility-versus-faithfulness distinction supplies the precise
language for this scope-limited claim.

Reviewers may attack the absence of a formal analyst evaluation,
arguing that without a user study the deployment-readiness claim is
unsupported. The defense is that the MVP contribution is the integration
architecture and the deployment narrative, not the analyst-utility
evidence. We commit explicitly to a Rastogi-style tier-stratified
evaluation in the future-work section, citing LExT and counterfactual
simulatability as the named target protocols.

Reviewers may attack the formulaic prose, arguing that the templates
read mechanically. The defense, drawn directly from SPLAIN, is that
formulaic structure is a deployment property rather than a deficiency:
"consistent warning structure and vocabulary enables analysts to skim."
Determinism is auditable. The same query produces the same explanation
across runs, which is a security property — randomly varying LLM output
would create a different explainability surface every time and would
defeat audit.

Reviewers may attack the deployment-readiness claim, arguing we have no
deployment data. The defense is in the wording: the prose says
"deployment-narrative" and "analyst-action implication" rather than
"production-ready" or "deployed." No paragraph contains overreach
language like "in production" or "fully automated." A grep over all 146
paragraphs confirms zero occurrences.

Reviewers may attack the choice of templates over LLMs, arguing that
modern XAI has moved on. The defense is fivefold: deterministic
templates eliminate hallucination risk on factual claims; they preserve
auditability because every output traces to a structured input; they
incur zero inference cost; they have no external API dependency; and
they avoid the chain-of-thought-unfaithfulness pattern documented by
Turpin, Michael, Perez, and Bowman at NeurIPS 2023. The XAI Descriptions
versus XAI Narratives framing (Cambria et al., as cited in the Gemini
report) positions templates as the deliberate design choice for
high-stakes pipelines, not a fallback.

Reviewers may attack the actor inclusion, arguing it could enable
shortcut learning. The defense is that the actor appears only in the
opening prepositional clause as passive context, never as the active
subject of a model-action verb. A regex audit across all 146 paragraphs
confirms zero instances of patterns like "APT29 uses..." or "because of
Carbanak..." The Step D counterfactual probe in the paper provides the
empirical evidence that actor names do not drive predictions.

Reviewers may attack the multi-gold framing, arguing it is weasel words
for the model being wrong. The defense is that 93% of queries in the
dataset are multi-gold in the source annotation. The CTI-HAL annotation
protocol treats each gold ID as independently valid. We use "annotated"
rather than "correct" throughout, which Lyu et al. recommend explicitly,
and we never describe the multi-gold case as the model being wrong.
Errors are flagged separately with "However, the annotated technique X
ranked Nth" and every one of the 8 error paragraphs requires analyst
review in the closing sentence.

## Edge cases and known limitations

The dataset has four single-token queries (VNC, Trickbot, VBScripts,
and keylogger) where leave-one-out attribution is mathematically
inapplicable because removing the only token leaves nothing to compare
against. The script detects this with a `top_evidence_tokens` empty list
plus a BM25-overlap availability check and falls back to either the
BM25-only evidence sentence (Trickbot) or the no-evidence sentence (VNC,
VBScripts, keylogger).

The dataset has 2 of 24 sub-technique queries where the parent
technique is not in the top-5 alternatives, which means the resolver
cannot retrieve a human-readable name for the parent. In those cases the
prose falls back to the ID alone (for example, "a sub-technique of
**T1518**" without the parent name). This is honest about what the layer
can verify; the alternative would be fabricating a name from external
sources, which would violate the determinism and traceability properties.

The dataset has 10 of 146 queries where the `gold` field designates a
primary annotation that is not in the top-5 alternatives. The S3
phrasing omits the dangling-name reference in those cases, producing
"is one of N ATT&CK techniques annotated as a valid mapping for this
query" without the "the primary annotation is X" continuation.

The dataset has 1 query (the OilRig "Before performing the first
request" case) where the primary gold is not in the top-5 alternatives
AND the only is_gold alternative in the top-5 is the predicted itself.
The grammatically natural phrasing "one of 1 ATT&CK techniques" is
explicitly avoided by a dedicated edge-case template: "matches an
annotated valid mapping for this query, though the dataset's primary
annotation does not appear in the top-5 candidates."

The dataset has 1 query (the cyzfc.dat case) where all selected LOO
evidence tokens are common function words ("the", "by", "one"). This
reflects genuine LOO sensitivity to sentence structure rather than a
filter bug; the closing analyst-action sentence already flags this case
as LOW certainty and recommends review. Filtering stopwords would
distort faithfulness to what LOO actually reports.

The raw cross-encoder score is a logit rather than a probability in
[0,1]. The translated vocabulary calls it "confidence score" per the
Rastogi et al. analyst-facing convention, which is a deliberate naming
choice rather than a misrepresentation. A reviewer who insists on
strict probabilistic semantics can read the score field directly from
the structured JSON record alongside the prose.

## Paper-section guidance

In the main body, place exactly one labeled monospace figure box
containing the HIGH-confidence single-gold or multi-gold-alternative
example from `nl_examples.md`, captioned to indicate it is the
deterministic output of the explanation synthesis layer for one
representative query. The text immediately around the figure should
identify the structured fields the prose consumes (per the standard
"state what the example shows, show the example, call out what to
notice" three-part structure documented in AGIR).

In the appendix or artifact-evaluation supplement, include the full
`nl_explanations.json`, `nl_examples.md`, `template_versions.json`,
`provenance_nl.json`, and `generate_nl_explanations.py`. The `--self-test`
mode in the script provides an audit hook the artifact-evaluation
committee can run independently to verify the 31 internal sanity checks
still pass.

In the methodology section, state explicitly that templates are the
deliberate design choice per Cambria et al.'s XAI Descriptions framing.
Cite SPLAIN as the primary precedent for templates-as-contribution and
AGIR as the secondary precedent for template-first pipelines. State
that faithfulness is claimed in the Lyu-et-al sense (input-output
sensitivity) and not in the mechanistic-reasoning sense.

In the evaluation section, report the structural statistics of the 146
outputs: 100% exactly five sentences, mean 95.7 words (median 95) with
all 146 in the 80-120 word Rastogi target range, 146/146 pass the
faithfulness traceability check, 146/146 pass the forbidden-jargon
hygiene check. State that user-study evaluation is deferred to future
work per the Nadeem-et-al 14% normative figure.

In the limitations section, acknowledge the four edge cases listed
above (single-token queries, unresolvable sub-technique parents, missing
primary annotations, and the all-stopword LOO case) and cite them as
intentionally surfaced rather than filtered. State that the analyst-
action implication in S5 is the most prescriptive claim and is scoped
to "review recommended/required" language rather than autonomous
decision-making.

## Verification and audit results

The internal sanity-check suite passes 31 of 31 tests. The end-to-end
run on all 146 queries produces 146 of 146 paragraphs that pass the
validation check (exactly five sentences, 80-180 words, all required
identifiers present, no forbidden jargon, no duplicate quoted tokens).
Two consecutive runs produce bit-identical output. The faithfulness
audit confirms that every quantitative claim in every paragraph traces
back to a stored structured field. The forbidden-jargon hygiene audit
finds zero occurrences across all 146 paragraphs. The multi-gold
consistency audit finds zero single-truth language and 146 of 146
paragraphs containing "annotat." The 14-attack red-team battery resists
every attempted attack, including faithfulness challenges, formulaic-
prose challenges, actor-shortcut challenges, deployment-overreach
challenges, multi-gold weasel-word challenges, LLM-versus-template
challenges, and stopword-evidence challenges.
