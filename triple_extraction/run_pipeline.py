from core.pipeline import Pipeline
from extractors.multi_agent_extractor import MultiAgentExtractor

"""
MAIN RUN SCRIPT
Execute the complete CTI extraction pipeline

Usage:
    python run_pipeline.py --text "APT29 used Cobalt Strike"
    python run_pipeline.py --file input.txt --output results.json
"""
from html import parser
import sys
import argparse
from pathlib import Path

# Add paths
sys.path.insert(0, '.')
sys.path.insert(0, 'core')
sys.path.insert(0, 'extractors')
sys.path.insert(0, 'validators')

from core.pipeline import Pipeline
from core.config import Config
from core.data_models import Triple
from extractors.multi_agent_extractor import MultiAgentExtractor

# Try to import optional components
try:
    from validators.stix_validator import STIXValidator
except:
    STIXValidator = None


def main():
    parser = argparse.ArgumentParser(description="CTI Modular Extraction Pipeline")
    parser.add_argument("--enable-stix", action="store_true", help="Enable STIX validator")
    parser.add_argument("--enable-markov", action="store_true", help="Enable Markov repairer")
    parser.add_argument("--text", type=str, help="Text to extract from")
    parser.add_argument("--file", type=str, help="File to extract from")
    parser.add_argument("--config", type=str, help="Config YAML file")
    parser.add_argument("--output", type=str, default="output/results.json", help="Output file")
    parser.add_argument("--model", type=str, help="Model name (overrides config)")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    
    # Load config
    config = Config(args.config) if args.config else Config()
    
    # Override model if specified
    if args.model:
        config.MODEL_NAME = args.model
    
    # Get text
    if args.file:
        with open(args.file) as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        # Default test
        text = """
        APT29, also known as Cozy Bear, is a sophisticated threat actor 
        attributed to Russian intelligence. They have been observed using 
        Cobalt Strike and Mimikatz to compromise government networks. 
        The group leverages spearphishing techniques to gain initial access.
        """
    
    print("\n" + "="*70)
    print("CTI MODULAR EXTRACTION PIPELINE")
    print("="*70)
    print(f"Model: {config.MODEL_NAME}")
    print(f"Text length: {len(text)} chars")
    print("="*70)
    
    # Build pipeline
    pipeline = Pipeline(config)
    
    # Add extractor
    pipeline.add_extractor(MultiAgentExtractor(config))
    
    # Add validator if available
    if STIXValidator:
        pipeline.add_validator(STIXValidator(config))
    
    # Run
    result = pipeline.run(text, debug=args.debug)
    
    # Display results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Total extracted: {result.statistics.get('total_extracted', 0)}")
    print(f"Valid:           {result.statistics.get('final_valid', 0)}")
    print(f"Invalid:         {result.statistics.get('final_invalid', 0)}")
    print(f"Time:            {result.statistics.get('total_time', 0):.2f}s")
    print("="*70)
    
    # Show valid triples
    if result.valid_triples:
        print("\n✓ VALID TRIPLES:")
        for i, triple in enumerate(result.valid_triples[:10], 1):
            print(f"  {i}. {triple.subject} ({triple.subject_type}) "
                  f"→ {triple.predicate} → "
                  f"{triple.object} ({triple.object_type})")
        
        if len(result.valid_triples) > 10:
            print(f"  ... and {len(result.valid_triples) - 10} more")
    
    # Show invalid triples
    if result.invalid_triples:
        print("\n✗ INVALID TRIPLES:")
        for i, triple in enumerate(result.invalid_triples[:5], 1):
            error = triple.metadata.get("validation_error", "unknown")
            print(f"  {i}. {triple.subject} → {triple.predicate} → {triple.object}")
            print(f"     Error: {error}")
    
    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pipeline.save_results(result, args.output)
    
    print(f"\n{'='*70}")
    print(f"✓ Complete! Results saved to {args.output}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
