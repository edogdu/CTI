"""Single source of truth for CTI pipeline ontology constants.

Uses native DNRTI/UCO entity types and predicates.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, FrozenSet, List, Set, Tuple

# =============================================================================
# CANONICAL DNRTI ENTITY TYPES (used by extraction prompt + training)
# =============================================================================

TYPES: Set[str] = {
    "APT", "MAL", "TOOL", "ACT", "IDTY", "LOC", "TIME",
    "FILE", "SECTEAM", "OS", "VULID", "VULNAME",
    "HASH", "DOM", "ENCR", "IP", "URL", "PROT", "EMAIL",
}

# =============================================================================
# CANONICAL PREDICATES (used by extraction prompt + training)
# =============================================================================

PREDS: Set[str] = {
    "uses", "usedBy", "targets", "targetedBy",
    "affiliatedWith", "associatedWith",
    "identifies", "identifiedBy", "monitors", "monitoredBy",
    "hasLocation", "hasAttackLocation", "hasAttackTime",
    "hasVulnerability", "contains",
}

# =============================================================================
# TYPE ALIASES — map common variations to canonical DNRTI types
# =============================================================================

TYPE_ALIASES: Dict[str, str] = {
    # Lowercase DNRTI labels → canonical
    "apt": "APT",
    "mal": "MAL",
    "tool": "TOOL",
    "act": "ACT",
    "idty": "IDTY",
    "loc": "LOC",
    "time": "TIME",
    "file": "FILE",
    "secteam": "SECTEAM",
    "os": "OS",
    "vulid": "VULID",
    "vulname": "VULNAME",
    "hash": "HASH",
    "dom": "DOM",
    "encr": "ENCR",
    "ip": "IP",
    "url": "URL",
    "prot": "PROT",
    "email": "EMAIL",
    # Common textual variants
    "threat-actor": "APT", "threatactor": "APT", "threatactors": "APT",
    "malware": "MAL", "backdoor": "MAL", "ransomware": "MAL", "trojan": "MAL",
    "attack-pattern": "ACT", "attackpattern": "ACT", "technique": "ACT", "campaign": "ACT",
    "identity": "IDTY", "organisation": "IDTY", "organization": "IDTY",
    "location": "LOC", "country": "LOC", "region": "LOC", "city": "LOC",
    "vulnerability": "VULNAME", "cve": "VULID",
    "domain": "DOM", "domainname": "DOM",
    "ipaddress": "IP", "ipv4": "IP", "ipv4addr": "IP", "ipv4-address": "IP",
    "ipv6": "IP", "ipv6addr": "IP", "ipv6-address": "IP",
    "emailaddress": "EMAIL",
    "protocol": "PROT",
    "encryption": "ENCR",
    "software": "TOOL", "product": "TOOL",
    "securityteam": "SECTEAM",
    "indicator": "IP",       # fallback for old STIX type
    "infrastructure": "PROT",  # fallback for old STIX type
    "intrusion-set": "APT", "intrusionset": "APT",
}

PRED_ALIASES: Dict[str, str] = {
    "use": "uses", "using": "uses",
    "target": "targets", "targeting": "targets",
    "exploit": "hasVulnerability", "exploits": "hasVulnerability",
    "attributed_to": "affiliatedWith",
    "communicates_with": "associatedWith", "communicateswith": "associatedWith",
    "originates_from": "hasLocation", "originatesfrom": "hasLocation",
    "located_at": "hasLocation",
    "usestechnique": "uses",
    "related-to": "associatedWith",
}

# =============================================================================
# DOMAIN / RANGE SCHEMA CONSTRAINTS (data-driven from full DNRTI dataset)
# =============================================================================

SCHEMA: Dict[str, Tuple[FrozenSet[str], FrozenSet[str]]] = {
    "affiliatedWith": (
        frozenset(["APT"]),
        frozenset(["APT"]),
    ),
    "associatedWith": (
        frozenset(TYPES),
        frozenset(TYPES),
    ),
    "contains": (
        frozenset(["DOM", "EMAIL", "FILE", "URL"]),
        frozenset(["DOM", "EMAIL", "FILE", "IP", "MAL", "URL"]),
    ),
    "hasAttackLocation": (
        frozenset(["ACT", "APT", "MAL"]),
        frozenset(["LOC"]),
    ),
    "hasAttackTime": (
        frozenset(["ACT", "APT", "MAL"]),
        frozenset(["TIME"]),
    ),
    "hasLocation": (
        frozenset(["IDTY", "SECTEAM"]),
        frozenset(["LOC"]),
    ),
    "hasVulnerability": (
        frozenset(["DOM", "FILE", "IDTY", "IP", "OS", "PROT", "TOOL"]),
        frozenset(["VULID", "VULNAME"]),
    ),
    "identifiedBy": (
        frozenset(["APT", "MAL"]),
        frozenset(["SECTEAM"]),
    ),
    "identifies": (
        frozenset(["SECTEAM"]),
        frozenset(["ACT", "APT", "MAL", "VULID", "VULNAME"]),
    ),
    "monitoredBy": (
        frozenset(["FILE", "IDTY", "LOC", "PROT"]),
        frozenset(["SECTEAM"]),
    ),
    "monitors": (
        frozenset(["SECTEAM"]),
        frozenset(["DOM", "FILE", "IDTY", "IP", "PROT", "URL"]),
    ),
    "targetedBy": (
        frozenset(["DOM", "IDTY", "LOC", "OS", "VULID", "VULNAME"]),
        frozenset(["ACT", "APT", "MAL"]),
    ),
    "targets": (
        frozenset(["ACT", "APT", "MAL"]),
        frozenset(["DOM", "IDTY", "OS", "VULID", "VULNAME"]),
    ),
    "usedBy": (
        frozenset(["ACT", "DOM", "EMAIL", "ENCR", "FILE", "HASH", "IP", "PROT", "TOOL", "URL"]),
        frozenset(["ACT", "APT", "DOM", "EMAIL", "FILE", "IDTY", "MAL", "OS", "TOOL", "URL"]),
    ),
    "uses": (
        frozenset(["ACT", "APT", "DOM", "EMAIL", "ENCR", "FILE", "IDTY", "IP", "MAL", "OS", "PROT", "SECTEAM", "TOOL", "URL"]),
        frozenset(["ACT", "DOM", "EMAIL", "ENCR", "FILE", "HASH", "IP", "MAL", "PROT", "SECTEAM", "TOOL", "URL"]),
    ),
}

# =============================================================================
# ENTITY ALIAS NORMALIZATION (threat actor name variants)
# =============================================================================

ENTITY_ALIASES: Dict[str, List[str]] = {
    "apt28": ["fancy bear", "sofacy", "pawn storm", "sednit", "strontium", "forest blizzard"],
    "apt29": ["cozy bear", "the dukes", "yttrium", "nobelium", "midnight blizzard"],
    "apt38": ["bluenoroff", "stardust chollima"],
    "apt41": ["double dragon", "barium", "winnti", "wicked panda"],
    "lazarus": ["lazarus group", "hidden cobra", "zinc", "labyrinth chollima"],
    "fin7": ["carbanak group", "carbon spider"],
    "carbanak": ["anunak", "carbanak group"],
    "emotet": ["geodo", "heodo"],
    "trickbot": ["trickloader"],
    "cobalt strike": ["cobaltstrike", "cs beacon", "beacon"],
}

ENTITY_ALIAS_MAP: Dict[str, str] = {}
for _canonical, _aliases in ENTITY_ALIASES.items():
    ENTITY_ALIAS_MAP[_canonical] = _canonical
    for _alias in _aliases:
        ENTITY_ALIAS_MAP[_alias.lower()] = _canonical

# =============================================================================
# DNRTI LABEL → STIX TYPE MAPPING (kept for evaluation notebook reverse-lookup)
# =============================================================================

DNRTI_LABEL_TO_STIX: Dict[str, str | None] = {
    # Old-format labels (from original notebook)
    "HackOrg": "threat-actor",
    "SamFile": "malware",
    "Tool": "tool",
    "Exp": "vulnerability",
    "Way": "attack-pattern",
    "Purp": "attack-pattern",
    "OffAct": "campaign",
    "Idus": "identity",
    "Area": "location",
    "Org": "identity",
    "Secteam": "identity",
    "Features": "indicator",
    "Time": None,
    # New-format labels (what our dataset actually uses)
    "APT": "threat-actor",
    "MAL": "malware",
    "IDTY": "identity",
    "ACT": "attack-pattern",
    "LOC": "location",
    "SECTEAM": "identity",
    "TOOL": "tool",
    "TIME": None,
    "FILE": "malware",
    "VULNAME": "vulnerability",
    "VULID": "vulnerability",
    "PROT": "infrastructure",
    "OS": "tool",
    "ENCR": "tool",
    "IP": "indicator",
    "EMAIL": "indicator",
    "HASH": "indicator",
    "DOM": "indicator",
}

STIX_TO_DNRTI_LABELS: Dict[str, List[str]] = defaultdict(list)
for _dnrti_label, _stix_type in DNRTI_LABEL_TO_STIX.items():
    if _stix_type is not None:
        STIX_TO_DNRTI_LABELS[_stix_type].append(_dnrti_label)

# =============================================================================
# SHARED TYPE/PREDICATE BLOCKS (used by prompt construction below)
# =============================================================================

_ENTITY_TYPE_BLOCK = (
    "ENTITY TYPES:\n"
    "- APT: Hacking groups, nation-state actors, threat actors "
    "(e.g., APT28, Lazarus Group, Fancy Bear)\n"
    "- MAL: Malicious software, backdoors, ransomware "
    "(e.g., Emotet, WannaCry, Stuxnet)\n"
    "- TOOL: Software tools used in attacks "
    "(e.g., Mimikatz, Cobalt Strike, PowerShell)\n"
    "- ACT: Offensive actions, attack techniques, campaigns "
    "(e.g., spear-phishing, credential dumping)\n"
    "- IDTY: Organizations, companies, agencies "
    "(e.g., Microsoft, US-CERT, NATO)\n"
    "- LOC: Countries, regions, cities "
    "(e.g., Russia, Middle East, Washington DC)\n"
    "- TIME: Dates, timestamps, time periods "
    "(e.g., August 2015, early 2017, Q3 2023)\n"
    "- FILE: Files used in attacks — executables, DLLs, documents, scripts "
    "(e.g., setup.exe, malicious.doc, loader.dll)\n"
    "- SECTEAM: Security teams, incident responders, researchers "
    "(e.g., FireEye, CrowdStrike, Unit 42)\n"
    "- OS: Operating systems "
    "(e.g., Windows, Linux, Android)\n"
    "- VULID: CVE identifiers "
    "(e.g., CVE-2021-44228, CVE-2017-0144)\n"
    "- VULNAME: Named vulnerabilities "
    "(e.g., EternalBlue, Log4Shell, Heartbleed)\n"
    "- HASH: File hashes — MD5, SHA256 "
    "(e.g., d41d8cd98f00b204e9800998ecf8427e)\n"
    "- DOM: Domain names "
    "(e.g., evil.com, c2.attacker.net)\n"
    "- ENCR: Encryption algorithms or tools "
    "(e.g., AES, RC4, RSA)\n"
    "- IP: IP addresses "
    "(e.g., 192.168.1.1, 10.0.0.1)\n"
    "- URL: Web addresses "
    "(e.g., http://evil.com/payload)\n"
    "- PROT: Network protocols "
    "(e.g., HTTP, SSH, DNS)\n"
    "- EMAIL: Email addresses "
    "(e.g., phisher@evil.com)\n"
)

_PREDICATE_BLOCK = (
    "PREDICATES:\n"
    "- uses: Actor/malware employs a tool, technique, or malware\n"
    "- usedBy: Tool/malware/technique is employed by an actor (inverse of uses)\n"
    "- targets: Actor/malware attacks an identity, OS, domain, or vulnerability\n"
    "- targetedBy: Identity/OS/domain is attacked by an actor (inverse of targets)\n"
    "- affiliatedWith: APT is linked to another APT group\n"
    "- associatedWith: General association between any two entities\n"
    "- identifies: Security team discovers or attributes an actor/malware/vulnerability\n"
    "- identifiedBy: Actor/malware is discovered by a security team (inverse of identifies)\n"
    "- monitors: Security team tracks a domain, file, or identity\n"
    "- monitoredBy: Entity is tracked by a security team (inverse of monitors)\n"
    "- hasLocation: Identity or security team is based in a location\n"
    "- hasAttackLocation: Attack or actor operates from a specific location\n"
    "- hasAttackTime: Attack or malware campaign occurred at a specific time\n"
    "- hasVulnerability: Entity exploits or is affected by a vulnerability\n"
    "- contains: Domain/file/URL contains or references another entity\n"
)

# =============================================================================
# EXTRACTION PROMPT (single-pass, joint NER+RE)
# =============================================================================

EXTRACTION_PROMPT = (
    "You are a cybersecurity threat intelligence analyst. "
    "Extract ALL entity-relationship triples from the text "
    "using the DNRTI ontology.\n\n"
    + _ENTITY_TYPE_BLOCK
    + "\n"
    + _PREDICATE_BLOCK
    + "\nRULES:\n"
    "1. Extract ONLY relationships explicitly stated in the text.\n"
    "2. Extract ALL valid triples. Do not stop early or truncate the list.\n"
    "3. Entity names must match the text exactly.\n"
    "4. Each entity must have a name and type from the lists above.\n"
    "5. Each predicate must be from the list above.\n"
    "6. Return NONE if no valid triples can be extracted.\n\n"
    "Text: {text}\n\n"
    "Output ONLY pipe-delimited triples, one per line:\n"
    "subject_name | subject_type | predicate | object_name | object_type\n\n"
    "Example:\n"
    "APT28 | APT | uses | Mimikatz | TOOL\n"
    "Emotet | MAL | targets | Microsoft | IDTY"
)

# =============================================================================
# NER PROMPT — Task 1: extract entities only
# =============================================================================

ENTITY_EXTRACTION_PROMPT = (
    "You are a cybersecurity threat intelligence analyst. "
    "Identify ALL named entities in the text and classify each "
    "using the DNRTI ontology.\n\n"
    + _ENTITY_TYPE_BLOCK
    + "\nRULES:\n"
    "1. Extract ONLY entities explicitly mentioned in the text.\n"
    "2. Entity names must match the text exactly.\n"
    "3. Each entity must have a type from the list above.\n"
    "4. Return NONE if no valid entities can be found.\n\n"
    "Text: {text}\n\n"
    "Output ONLY pipe-delimited entities, one per line:\n"
    "entity_name | entity_type\n\n"
    "Example:\n"
    "APT28 | APT\n"
    "Mimikatz | TOOL\n"
    "August 2015 | TIME"
)

# =============================================================================
# RE PROMPT — Task 2: classify relations given known entities
# =============================================================================

RELATION_EXTRACTION_PROMPT = (
    "You are a cybersecurity threat intelligence analyst. "
    "Given the identified entities below, extract ALL relationships "
    "between them using the DNRTI ontology.\n\n"
    "IDENTIFIED ENTITIES:\n{entity_hints}\n\n"
    + _PREDICATE_BLOCK
    + "\nRULES:\n"
    "1. Extract ONLY relationships explicitly stated in the text.\n"
    "2. Use ONLY the entities listed above as subjects and objects.\n"
    "3. Entity names and types must match the list above exactly.\n"
    "4. Each predicate must be from the predicate list above.\n"
    "5. Return NONE if no valid relationships exist "
    "between the listed entities.\n\n"
    "Text: {text}\n\n"
    "Output ONLY pipe-delimited triples, one per line:\n"
    "subject_name | subject_type | predicate | object_name | object_type\n\n"
    "Example:\n"
    "APT28 | APT | uses | Mimikatz | TOOL\n"
    "Emotet | MAL | targets | Microsoft | IDTY"
)


def format_entity_hints(entities: list[dict[str, str]]) -> str:
    """Format entity list as hint text for the RE prompt.

    Each entity dict has 'name' and 'type' keys.
    Returns one line per entity: '- EntityName (ENTITY_TYPE)'
    """
    if not entities:
        return "(No entities identified)"
    lines = []
    for ent in entities:
        lines.append(f"- {ent['name']} ({ent['type']})")
    return "\n".join(lines)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def canon_type(t: str) -> str:
    """Canonicalize an entity type string to a known DNRTI type."""
    if not isinstance(t, str):
        return str(t) if t is not None else ""
    t_stripped = t.strip()
    # Already a canonical type
    if t_stripped in TYPES:
        return t_stripped
    # Try uppercase (e.g. "apt" → "APT")
    if t_stripped.upper() in TYPES:
        return t_stripped.upper()
    # Try alias lookup (lowercase, stripped of punctuation)
    t_lower = t_stripped.lower().replace(" ", "").replace("_", "").replace("-", "")
    return TYPE_ALIASES.get(t_lower, t_stripped)


def canon_pred(p: str) -> str:
    """Canonicalize a predicate string to a known predicate."""
    if not isinstance(p, str):
        return str(p) if p is not None else ""
    s = p.strip().lower().replace(" ", "_")
    s = PRED_ALIASES.get(s, s)
    return s
