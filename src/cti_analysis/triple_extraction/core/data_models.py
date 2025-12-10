"""
CORE DATA MODELS
From your CTI_Entity_Extraction.ipynb

File: core/data_models.py
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional


# ============================================================================
# ENTITY AND TRIPLE (From Notebook Cell 5)
# ============================================================================

@dataclass
class Entity:
    """Entity extracted from text (from your notebook)."""
    text: str
    type: str
    confidence: float = 0.95
    
    def to_dict(self):
        return {
            "text": self.text,
            "type": self.type,
            "confidence": self.confidence
        }


@dataclass
class Triple:
    """Relation triple (from your notebook)."""
    subject: str
    subject_type: str
    predicate: str
    object: str
    object_type: str
    confidence: float = 0.95
    evidence: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_tuple(self) -> Tuple[str, str, str]:
        """Convert to (subject, predicate, object) tuple."""
        return (
            self.subject.lower(),
            self.predicate.lower(),
            self.object.lower()
        )
    
    def to_dict(self):
        return {
            "subject": {"name": self.subject, "type": self.subject_type},
            "predicate": self.predicate,
            "object": {"name": self.object, "type": self.object_type},
            "confidence": self.confidence,
            "evidence": self.evidence,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Create from dictionary."""
        subject = data.get("subject", {})
        obj = data.get("object", {})
        
        return cls(
            subject=subject.get("name", ""),
            subject_type=subject.get("type", ""),
            predicate=data.get("predicate", ""),
            object=obj.get("name", ""),
            object_type=obj.get("type", ""),
            confidence=data.get("confidence", 0.95),
            evidence=data.get("evidence", ""),
            metadata=data.get("metadata", {})
        )


# ============================================================================
# EXTRACTION RESULT
# ============================================================================

@dataclass
class ExtractionResult:
    """Results from extraction pipeline."""
    triples: List[Triple] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    valid_triples: List[Triple] = field(default_factory=list)
    invalid_triples: List[Triple] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def add_triple(self, triple: Triple, is_valid: bool = True):
        """Add a triple to results."""
        self.triples.append(triple)
        if is_valid:
            self.valid_triples.append(triple)
        else:
            self.invalid_triples.append(triple)
    
    def to_dict(self):
        return {
            "triples": [t.to_dict() for t in self.triples],
            "entities": [e.to_dict() for e in self.entities],
            "valid_triples": [t.to_dict() for t in self.valid_triples],
            "invalid_triples": [t.to_dict() for t in self.invalid_triples],
            "statistics": self.statistics,
            "errors": self.errors
        }


# ============================================================================
# EVALUATION RESULT
# ============================================================================

@dataclass
class EvaluationResult:
    """Results from evaluation."""
    dataset: str
    samples: int
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    hit_rate: float
    per_sample: List[Dict] = field(default_factory=list)
    
    def to_dict(self):
        return {
            "dataset": self.dataset,
            "samples": self.samples,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "hit_rate": self.hit_rate,
            "per_sample": self.per_sample
        }


# ============================================================================
# ONTOLOGY DEFINITIONS (From Notebook Cell 5)
# ============================================================================

STIX_ENTITY_TYPES = {
    "malware", "tool", "attack-pattern", "threat-actor", "intrusion-set",
    "infrastructure", "campaign", "indicator", "vulnerability", "identity",
    "tactic", "technique", "sub-technique", "c2", "domain", "ip",
    "organization", "sector", "country", "location"
}

STIX_REL_TYPES = {
    "uses", "delivers", "drops", "downloads", "installs", "exploits",
    "targets", "attributed-to", "mitigates", "indicates", "based-on",
    "derived-from", "communicates-with", "hosts", "located-at",
    "originates-from", "executes", "contains", "related-to"
}

# MALONT classes (from your extraction.py)
MALONT_CLASSES = {
    'Staging', 'Adware', 'CommandAndControl', 'Spyware', 'DDoS', 'DomainName',
    'Dropper', 'Port', 'MD5', 'Protocol', 'VirusScanner', 'Downloader',
    'Ransomware', 'OperatingSystem', 'Rootkit', 'AttackPattern_SmallDescription',
    'IPAddress', 'Bootkit', 'Hardware', 'SSDeep', 'Application', 'AttackPattern',
    'Phishing', 'Campaign', 'SHA-256', 'System', 'Vulnerability_Desc',
    'Anonymization', 'Backdoor', 'Location', 'Organization', 'Reconnaissance',
    'Exploit-kit', 'Time', 'MalwareAnalysis', 'ResourceExploitation', 'SHA',
    'HostingMalware', 'SHA-1', 'Unknown', 'HostingTargetLists', 'Hash',
    'AttackPattern_LargeDescription', 'Software', 'Network', 'Indicator',
    'Trojan', 'Botnet', 'Worm', 'EmailAddress', 'Malware', 'RogueSecuritySoftware',
    'vHash', 'Filepath', 'Region', 'Report', 'Virus', 'ThreatActor', 'Keylogger',
    'Browser', 'ScreenCapture', 'Vulnerability_CVEID', 'URL', 'Wiper', 'Filename',
    'Infrastructure', 'MalwareFamily', 'Person', 'Webshell', 'Vulnerability',
    'Bot', 'RemoteAccessTrojan-RAT', 'Country', 'Exfiltration', 'Amplification'
}

MALONT_PREDICATES = {
    "targets", "communicatesWith", "uses", "has", "hasAlias",
    "hasVulnerability", "indicates", "exploits", "hasAuthor", "belongsTo"
}


# ============================================================================
# NORMALIZATION HELPERS (From Notebook)
# ============================================================================

def canonicalize_entity(entity_type: str) -> str:
    """Normalize entity type (from notebook)."""
    if not entity_type:
        return "unknown"
    
    # Convert to lowercase, remove spaces
    norm = entity_type.lower().strip().replace(" ", "-").replace("_", "-")
    
    # Map common aliases
    aliases = {
        "apt": "threat-actor",
        "group": "threat-actor",
        "actor": "threat-actor",
        "malware-family": "malware",
        "rat": "tool",
        "trojan": "malware",
        "ransomware": "malware",
        "exploit": "attack-pattern",
        "technique": "attack-pattern",
        "org": "organization",
        "company": "organization",
    }
    
    if norm in aliases:
        return aliases[norm]
    
    return norm


def normalize_predicate(predicate: str) -> str:
    """Normalize predicate (from notebook)."""
    if not predicate:
        return "related-to"
    
    # Convert to lowercase, use hyphens
    norm = predicate.lower().strip().replace("_", "-").replace(" ", "-")
    
    # Map common aliases
    aliases = {
        "use": "uses",
        "using": "uses",
        "utilized": "uses",
        "utilize": "uses",
        "target": "targets",
        "targeting": "targets",
        "exploit": "exploits",
        "exploiting": "exploits",
        "deliver": "delivers",
        "delivering": "delivers",
        "drop": "drops",
        "dropping": "drops",
    }
    
    if norm in aliases:
        return aliases[norm]
    
    return norm


if __name__ == "__main__":
    print("CTI Data Models")
    print("=" * 60)
    print(f"STIX Entity Types: {len(STIX_ENTITY_TYPES)}")
    print(f"STIX Relation Types: {len(STIX_REL_TYPES)}")
    print(f"MALONT Classes: {len(MALONT_CLASSES)}")
    print(f"MALONT Predicates: {len(MALONT_PREDICATES)}")
    
    # Test
    triple = Triple(
        subject="APT29",
        subject_type="threat-actor",
        predicate="uses",
        object="Cobalt Strike",
        object_type="tool",
        confidence=0.95
    )
    print(f"\nTest Triple: {triple.to_tuple()}")
    print(f"Dict: {triple.to_dict()}")
