"""
MULTI-AGENT EXTRACTOR
From your notebook Cell 9 - EntityAgent + RelationshipAgent

This is your PRIMARY extraction method!

File: extractors/multi_agent_extractor.py
"""
import sys
sys.path.insert(0, '..')

from core.pipeline import Extractor
from core.data_models import Entity, Triple, ExtractionResult, STIX_ENTITY_TYPES, STIX_REL_TYPES
from core.utils import generate_response, repair_json
from core.config import Config

import json
from typing import List


# ============================================================================
# ENTITY AGENT (From Notebook Cell 9)
# ============================================================================

class EntityAgent:
    """
    Extract entities from text.
    From your notebook Cell 9.
    """
    
    PROMPT_TEMPLATE = """Extract CTI entities from TEXT.
Allowed types: malware, tool, attack-pattern, threat-actor, intrusion-set, indicator, vulnerability, org, tactic.
Return ONLY JSON:
{"entities":[{"text":"...","type":"malware|tool|attack-pattern|threat-actor|intrusion-set|indicator|vulnerability|org|tactic","confidence":0.9}, ...]}
TEXT:
"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
    
    def run(self, text: str, debug: bool = False) -> List[Entity]:
        """Extract entities from text."""
        prompt = self.PROMPT_TEMPLATE + text + "\nOUTPUT:"
        
        response = generate_response(
            prompt,
            model=self.config.MODEL_NAME,
            base_url=self.config.OLLAMA_BASE_URL,
            max_tokens=700,
            temperature=self.config.TEMPERATURE,
            debug=debug
        )
        
        try:
            data = json.loads(repair_json(response))
            entities = []
            
            for e in data.get("entities", []):
                if e.get("text") and e.get("type"):
                    entity = Entity(
                        text=e["text"],
                        type=str(e["type"]).lower().strip(),
                        confidence=float(e.get("confidence", 0.9))
                    )
                    entities.append(entity)
            
            return entities
        
        except Exception as e:
            if debug:
                print(f"[EntityAgent Error] {e}")
            return []


# ============================================================================
# RELATIONSHIP AGENT (From Notebook Cell 9)
# ============================================================================

class RelationshipAgent:
    """
    Extract relationships between entities.
    From your notebook Cell 9.
    """
    
    PROMPT_TEMPLATE = """Extract CTI relationships from TEXT.
Entities: {entities}
Allowed predicates: uses, delivers, drops, downloads, installs, exploits, targets, attributed-to, mitigates, indicates, executes.
Return ONLY JSON:
{"relationships":[{"subject":"...","predicate":"uses|delivers|drops|downloads|installs|exploits|targets|attributed-to|mitigates|indicates|executes","object":"...","confidence":0.9}, ...]}
TEXT:
"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
    
    def run(self, text: str, entities: List[Entity], debug: bool = False) -> List[Triple]:
        """Extract relationships from text given entities."""
        # Format entities for prompt
        entity_str = ", ".join([f"{e.text} ({e.type})" for e in entities])
        
        prompt = self.PROMPT_TEMPLATE.format(entities=entity_str) + text + "\nOUTPUT:"
        
        response = generate_response(
            prompt,
            model=self.config.MODEL_NAME,
            base_url=self.config.OLLAMA_BASE_URL,
            max_tokens=900,
            temperature=self.config.TEMPERATURE,
            debug=debug
        )
        
        try:
            data = json.loads(repair_json(response))
            triples = []
            
            # Build entity lookup
            entity_dict = {e.text.lower(): e for e in entities}
            
            for r in data.get("relationships", []):
                subj_text = r.get("subject", "")
                obj_text = r.get("object", "")
                pred = r.get("predicate", "")
                
                if not (subj_text and obj_text and pred):
                    continue
                
                # Find entity types
                subj_entity = entity_dict.get(subj_text.lower())
                obj_entity = entity_dict.get(obj_text.lower())
                
                subj_type = subj_entity.type if subj_entity else "unknown"
                obj_type = obj_entity.type if obj_entity else "unknown"
                
                triple = Triple(
                    subject=subj_text,
                    subject_type=subj_type,
                    predicate=pred.lower().strip(),
                    object=obj_text,
                    object_type=obj_type,
                    confidence=float(r.get("confidence", 0.9)),
                    evidence=text[:200]  # First 200 chars as evidence
                )
                
                triples.append(triple)
            
            return triples
        
        except Exception as e:
            if debug:
                print(f"[RelationshipAgent Error] {e}")
            return []


# ============================================================================
# MULTI-AGENT EXTRACTOR
# ============================================================================

class MultiAgentExtractor(Extractor):
    """
    Multi-agent extraction combining EntityAgent + RelationshipAgent.
    This is your PRIMARY extraction method from the notebook!
    """
    
    def __init__(self, config: Config = None):
        super().__init__("multi_agent_extractor", config)
        self.config = config or Config()
        self.entity_agent = EntityAgent(self.config)
        self.relationship_agent = RelationshipAgent(self.config)
    
    def extract(self, text: str) -> ExtractionResult:
        """
        Extract entities and relationships using multi-agent approach.
        
        Steps:
        1. EntityAgent extracts entities
        2. RelationshipAgent extracts relationships between entities
        """
        result = ExtractionResult()
        
        # Step 1: Extract entities
        entities = self.entity_agent.run(text, debug=False)
        result.entities = entities
        
        if not entities:
            return result
        
        # Step 2: Extract relationships
        triples = self.relationship_agent.run(text, entities, debug=False)
        result.triples = triples
        
        return result


if __name__ == "__main__":
    print("Testing Multi-Agent Extractor...")
    print("=" * 60)
    
    # Test
    config = Config()
    extractor = MultiAgentExtractor(config)
    
    text = """
    APT29, also known as Cozy Bear, is a sophisticated threat actor attributed
    to Russian intelligence. They have been observed using Cobalt Strike and
    Mimikatz to compromise government networks. The group leverages spearphishing
    techniques to gain initial access.
    """
    
    print(f"Input text: {text[:100]}...")
    print("\nExtracting...")
    
    result = extractor.extract(text)
    
    print(f"\nEntities found: {len(result.entities)}")
    for e in result.entities[:5]:
        print(f"  - {e.text} ({e.type})")
    
    print(f"\nTriples found: {len(result.triples)}")
    for t in result.triples[:5]:
        print(f"  - {t.subject} → {t.predicate} → {t.object}")
    
    print("\n✓ Multi-agent extractor ready!")
