"""
CONFIGURATION MANAGEMENT
From your CTI_Entity_Extraction.ipynb Config class

File: core/config.py
"""
import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any


class Config:
    """
    Configuration for CTI extraction pipeline.
    Based on your notebook Cell 3 Config class.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        # Dataset directories
        self.CTI_HAL_DIR = os.getenv("CTI_HAL_DIR", "/content/CTI-HAL")
        self.ANNO_CTR_DIR = os.getenv("ANNO_CTR_DIR", "/content/ANNO-CTR")
        self.DNRTI_DIR = os.getenv("DNRTI_DIR", "/content/DNRTI")
        
        # Model configuration
        self.MODEL_NAME = os.getenv("MODEL_NAME", "gemma2:9b")
        self.OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
        self.MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1200"))
        
        # Evaluation configuration
        self.USE_VECTOR_SIMILARITY = True
        self.SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.66"))
        self.DEBUG_FIRST = int(os.getenv("DEBUG_FIRST", "2"))
        
        # Chunking configuration
        self.ENABLE_SEMANTIC_CHUNKING = True
        self.CHUNK_WINDOW_TOKENS = int(os.getenv("CHUNK_WINDOW_TOKENS", "600"))
        self.SENT_OVERLAP = int(os.getenv("SENT_OVERLAP", "2"))
        self.SIM_MERGE_THRESHOLD = float(os.getenv("SIM_MERGE_THRESHOLD", "0.78"))
        
        # Post-processing (from notebook defaults)
        self.ENABLE_MULTI_AGENT = True
        self.ENABLE_CONSENSUS = True
        self.ENABLE_MARKOV_SMOOTHING = True
        self.ENABLE_SZF = True
        self.ENABLE_PRONOUN_BACKFILL = True
        
        # Consensus parameters (from your consensus.py)
        self.CONSENSUS_M = int(os.getenv("CONSENSUS_M", "2"))
        self.CONSENSUS_TAU_NAME = float(os.getenv("CONSENSUS_TAU_NAME", "0.90"))
        self.NUM_PROMPTS = int(os.getenv("NUM_PROMPTS", "3"))
        
        # Markov smoothing parameters
        self.MARKOV_ALPHA = float(os.getenv("MARKOV_ALPHA", "0.1"))
        
        # Performance
        self.PARALLEL_EVAL = False
        self.GENERATE_PLOTS = False
        
        # Load from YAML if provided
        if config_file and os.path.exists(config_file):
            self.load_from_yaml(config_file)
    
    def load_from_yaml(self, config_file: str):
        """Load configuration from YAML file."""
        with open(config_file, 'r') as f:
            data = yaml.safe_load(f)
        
        # Update from YAML
        if 'model' in data:
            self.MODEL_NAME = data['model'].get('name', self.MODEL_NAME)
            self.OLLAMA_BASE_URL = data['model'].get('ollama_url', self.OLLAMA_BASE_URL)
            self.TEMPERATURE = data['model'].get('temperature', self.TEMPERATURE)
            self.MAX_TOKENS = data['model'].get('max_tokens', self.MAX_TOKENS)
        
        if 'chunking' in data:
            self.ENABLE_SEMANTIC_CHUNKING = data['chunking'].get('enable', self.ENABLE_SEMANTIC_CHUNKING)
            self.CHUNK_WINDOW_TOKENS = data['chunking'].get('max_tokens', self.CHUNK_WINDOW_TOKENS)
            self.SENT_OVERLAP = data['chunking'].get('overlap_sentences', self.SENT_OVERLAP)
        
        if 'processing' in data:
            self.ENABLE_MULTI_AGENT = data['processing'].get('enable_multi_agent', self.ENABLE_MULTI_AGENT)
            self.ENABLE_CONSENSUS = data['processing'].get('enable_consensus', self.ENABLE_CONSENSUS)
            self.ENABLE_MARKOV_SMOOTHING = data['processing'].get('enable_markov', self.ENABLE_MARKOV_SMOOTHING)
            self.ENABLE_SZF = data['processing'].get('enable_szf', self.ENABLE_SZF)
        
        if 'evaluation' in data:
            self.USE_VECTOR_SIMILARITY = data['evaluation'].get('use_vector_similarity', self.USE_VECTOR_SIMILARITY)
            self.SIMILARITY_THRESHOLD = data['evaluation'].get('threshold', self.SIMILARITY_THRESHOLD)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model": {
                "name": self.MODEL_NAME,
                "ollama_url": self.OLLAMA_BASE_URL,
                "temperature": self.TEMPERATURE,
                "max_tokens": self.MAX_TOKENS
            },
            "chunking": {
                "enable": self.ENABLE_SEMANTIC_CHUNKING,
                "max_tokens": self.CHUNK_WINDOW_TOKENS,
                "overlap_sentences": self.SENT_OVERLAP,
                "sim_merge_threshold": self.SIM_MERGE_THRESHOLD
            },
            "processing": {
                "enable_multi_agent": self.ENABLE_MULTI_AGENT,
                "enable_consensus": self.ENABLE_CONSENSUS,
                "enable_markov": self.ENABLE_MARKOV_SMOOTHING,
                "enable_szf": self.ENABLE_SZF,
                "enable_pronoun": self.ENABLE_PRONOUN_BACKFILL
            },
            "consensus": {
                "m": self.CONSENSUS_M,
                "tau_name": self.CONSENSUS_TAU_NAME,
                "num_prompts": self.NUM_PROMPTS
            },
            "evaluation": {
                "use_vector_similarity": self.USE_VECTOR_SIMILARITY,
                "threshold": self.SIMILARITY_THRESHOLD,
                "debug_first": self.DEBUG_FIRST
            }
        }
    
    def save_to_yaml(self, output_file: str):
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        print(f"Config saved to {output_file}")
    
    def __repr__(self):
        return f"Config(model={self.MODEL_NAME}, chunking={self.ENABLE_SEMANTIC_CHUNKING}, multi_agent={self.ENABLE_MULTI_AGENT})"


# Global config instance
config = Config()


def load_config(config_file: Optional[str] = None) -> Config:
    """Load configuration."""
    return Config(config_file)


if __name__ == "__main__":
    print("CTI Pipeline Configuration")
    print("=" * 60)
    
    config = Config()
    print(config)
    print("\nConfiguration:")
    import json
    print(json.dumps(config.to_dict(), indent=2))
    
    # Test save/load
    config.save_to_yaml("config/test.yaml")
    print("\nSaved to config/test.yaml")
