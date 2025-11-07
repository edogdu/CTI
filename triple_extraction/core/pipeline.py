"""
MAIN PIPELINE ORCHESTRATOR
Integrates all components into a unified extraction pipeline

File: core/pipeline.py
"""
import sys
sys.path.insert(0, '.')

from typing import List, Optional, Any
import time
import logging
from pathlib import Path

from core.data_models import Triple, Entity, ExtractionResult
from core.config import Config

logger = logging.getLogger(__name__)


# ============================================================================
# COMPONENT BASE CLASSES
# ============================================================================

class PipelineComponent:
    """Base class for all pipeline components."""
    
    def __init__(self, name: str, config: Optional[Config] = None):
        self.name = name
        self.config = config or Config()
        self.enabled = True
        self.statistics = {}
    
    def process(self, data: Any) -> Any:
        """Process data through this component."""
        raise NotImplementedError
    
    def setup(self):
        """Optional setup before processing."""
        pass
    
    def teardown(self):
        """Optional cleanup after processing."""
        pass


class Extractor(PipelineComponent):
    """Base class for extractors."""
    
    def extract(self, text: str) -> ExtractionResult:
        """Extract triples from text."""
        raise NotImplementedError
    
    def process(self, text: str) -> ExtractionResult:
        return self.extract(text)


class Processor(PipelineComponent):
    """Base class for processors."""
    
    def process_triples(self, triples: List[Triple]) -> List[Triple]:
        """Process triples."""
        raise NotImplementedError
    
    def process(self, result: ExtractionResult) -> ExtractionResult:
        result.triples = self.process_triples(result.triples)
        if result.valid_triples:
            result.valid_triples = self.process_triples(result.valid_triples)
        return result


class Filter(PipelineComponent):
    """Base class for filters."""
    
    def filter_triples(self, triples: List[Triple]) -> tuple[List[Triple], List[Triple]]:
        """Filter triples, return (kept, filtered)."""
        raise NotImplementedError
    
    def process(self, result: ExtractionResult) -> ExtractionResult:
        kept, filtered = self.filter_triples(result.triples)
        result.triples = kept
        result.valid_triples = kept
        result.invalid_triples.extend(filtered)
        return result


class Validator(PipelineComponent):
    """Base class for validators."""
    
    def validate_triple(self, triple: Triple) -> tuple[bool, Optional[str]]:
        """Validate a triple."""
        raise NotImplementedError
    
    def process(self, result: ExtractionResult) -> ExtractionResult:
        valid, invalid = [], []
        for triple in result.triples:
            is_valid, error = self.validate_triple(triple)
            if is_valid:
                valid.append(triple)
            else:
                if error:
                    triple.metadata["validation_error"] = error
                invalid.append(triple)
        
        result.valid_triples = valid
        result.invalid_triples.extend(invalid)
        return result


class Repairer(PipelineComponent):
    """Base class for repairers."""
    
    def repair_triple(self, triple: Triple) -> Optional[Triple]:
        """Repair an invalid triple."""
        raise NotImplementedError
    
    def process(self, result: ExtractionResult) -> ExtractionResult:
        repaired_valid = []
        still_invalid = []
        
        for triple in result.invalid_triples:
            repaired = self.repair_triple(triple)
            if repaired:
                repaired.metadata["repaired"] = True
                repaired_valid.append(repaired)
            else:
                still_invalid.append(triple)
        
        result.valid_triples.extend(repaired_valid)
        result.invalid_triples = still_invalid
        return result


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class Pipeline:
    """
    Main extraction pipeline orchestrator.
    Manages flow through extractors → processors → filters → validators → repairers.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        
        # Component lists
        self.extractors: List[Extractor] = []
        self.processors: List[Processor] = []
        self.filters: List[Filter] = []
        self.validators: List[Validator] = []
        self.repairers: List[Repairer] = []
        
        # Tracking
        self.execution_log: List[dict] = []
    
    # === Component Registration ===
    
    def add_extractor(self, extractor: Extractor):
        """Add an extractor."""
        self.extractors.append(extractor)
        return self
    
    def add_processor(self, processor: Processor):
        """Add a processor."""
        self.processors.append(processor)
        return self
    
    def add_filter(self, filter_: Filter):
        """Add a filter."""
        self.filters.append(filter_)
        return self
    
    def add_validator(self, validator: Validator):
        """Add a validator."""
        self.validators.append(validator)
        return self
    
    def add_repairer(self, repairer: Repairer):
        """Add a repairer."""
        self.repairers.append(repairer)
        return self
    
    # === Pipeline Execution ===
    
    def run(self, text: str, debug: bool = False) -> ExtractionResult:
        """
        Run complete pipeline on text.
        
        Flow:
        1. Extract triples
        2. Process (normalize, enrich)
        3. Filter (consensus, confidence)
        4. Validate (schema, ontology)
        5. Repair (fix invalid)
        """
        start_time = time.time()
        self.execution_log = []
        
        if debug:
            logger.setLevel(logging.DEBUG)
            print(f"\n{'='*70}")
            print("PIPELINE START")
            print('='*70)
        
        # Stage 1: Extract
        result = self._run_extractors(text, debug)
        
        # Stage 2: Process
        result = self._run_processors(result, debug)
        
        # Stage 3: Filter
        result = self._run_filters(result, debug)
        
        # Stage 4: Validate
        result = self._run_validators(result, debug)
        
        # Stage 5: Repair
        result = self._run_repairers(result, debug)
        
        # Finalize
        elapsed = time.time() - start_time
        result.statistics.update({
            "total_time": elapsed,
            "total_extracted": len(result.triples),
            "final_valid": len(result.valid_triples),
            "final_invalid": len(result.invalid_triples)
        })
        
        if debug:
            print(f"\n{'='*70}")
            print(f"PIPELINE COMPLETE ({elapsed:.2f}s)")
            print(f"  Valid:   {len(result.valid_triples)}")
            print(f"  Invalid: {len(result.invalid_triples)}")
            print('='*70 + "\n")
        
        return result
    
    def _run_extractors(self, text: str, debug: bool) -> ExtractionResult:
        """Run all extractors."""
        if not self.extractors:
            raise ValueError("No extractors registered")
        
        combined = ExtractionResult()
        
        for extractor in self.extractors:
            if not extractor.enabled:
                continue
            
            start = time.time()
            if debug:
                print(f"\n[STAGE 1: EXTRACTION]")
                print(f"  Running: {extractor.name}")
            
            try:
                result = extractor.extract(text)
                combined.triples.extend(result.triples)
                combined.entities.extend(result.entities)
                
                self.execution_log.append({
                    "stage": "extraction",
                    "component": extractor.name,
                    "time": time.time() - start,
                    "triples": len(result.triples)
                })
                
                if debug:
                    print(f"  ✓ Extracted: {len(result.triples)} triples in {time.time()-start:.2f}s")
            
            except Exception as e:
                logger.error(f"Extractor '{extractor.name}' failed: {e}")
                combined.errors.append(f"{extractor.name}: {str(e)}")
                if debug:
                    print(f"  ✗ Error: {e}")
        
        return combined
    
    def _run_processors(self, result: ExtractionResult, debug: bool) -> ExtractionResult:
        """Run all processors."""
        if debug and self.processors:
            print(f"\n[STAGE 2: PROCESSING]")
        
        for processor in self.processors:
            if not processor.enabled:
                continue
            
            start = time.time()
            if debug:
                print(f"  Running: {processor.name}")
            
            try:
                result = processor.process(result)
                
                self.execution_log.append({
                    "stage": "processing",
                    "component": processor.name,
                    "time": time.time() - start
                })
                
                if debug:
                    print(f"  ✓ Processed in {time.time()-start:.2f}s")
            
            except Exception as e:
                logger.error(f"Processor '{processor.name}' failed: {e}")
                result.errors.append(f"{processor.name}: {str(e)}")
                if debug:
                    print(f"  ✗ Error: {e}")
        
        return result
    
    def _run_filters(self, result: ExtractionResult, debug: bool) -> ExtractionResult:
        """Run all filters."""
        if debug and self.filters:
            print(f"\n[STAGE 3: FILTERING]")
        
        for filter_ in self.filters:
            if not filter_.enabled:
                continue
            
            start = time.time()
            before = len(result.triples)
            
            if debug:
                print(f"  Running: {filter_.name}")
            
            try:
                result = filter_.process(result)
                after = len(result.triples)
                
                self.execution_log.append({
                    "stage": "filtering",
                    "component": filter_.name,
                    "time": time.time() - start,
                    "filtered": before - after
                })
                
                if debug:
                    print(f"  ✓ {before} → {after} triples (filtered {before-after})")
            
            except Exception as e:
                logger.error(f"Filter '{filter_.name}' failed: {e}")
                result.errors.append(f"{filter_.name}: {str(e)}")
                if debug:
                    print(f"  ✗ Error: {e}")
        
        return result
    
    def _run_validators(self, result: ExtractionResult, debug: bool) -> ExtractionResult:
        """Run all validators."""
        if debug and self.validators:
            print(f"\n[STAGE 4: VALIDATION]")
        
        for validator in self.validators:
            if not validator.enabled:
                continue
            
            start = time.time()
            
            if debug:
                print(f"  Running: {validator.name}")
            
            try:
                result = validator.process(result)
                
                self.execution_log.append({
                    "stage": "validation",
                    "component": validator.name,
                    "time": time.time() - start,
                    "invalid": len(result.invalid_triples)
                })
                
                if debug:
                    print(f"  ✓ Found {len(result.invalid_triples)} invalid triples")
            
            except Exception as e:
                logger.error(f"Validator '{validator.name}' failed: {e}")
                result.errors.append(f"{validator.name}: {str(e)}")
                if debug:
                    print(f"  ✗ Error: {e}")
        
        return result
    
    def _run_repairers(self, result: ExtractionResult, debug: bool) -> ExtractionResult:
        """Run all repairers."""
        if debug and self.repairers:
            print(f"\n[STAGE 5: REPAIR]")
        
        for repairer in self.repairers:
            if not repairer.enabled:
                continue
            
            start = time.time()
            before = len(result.invalid_triples)
            
            if debug:
                print(f"  Running: {repairer.name}")
            
            try:
                result = repairer.process(result)
                repaired = before - len(result.invalid_triples)
                
                self.execution_log.append({
                    "stage": "repair",
                    "component": repairer.name,
                    "time": time.time() - start,
                    "repaired": repaired
                })
                
                if debug:
                    print(f"  ✓ Repaired {repaired} triples")
            
            except Exception as e:
                logger.error(f"Repairer '{repairer.name}' failed: {e}")
                result.errors.append(f"{repairer.name}: {str(e)}")
                if debug:
                    print(f"  ✗ Error: {e}")
        
        return result
    
    # === Utility Methods ===
    
    def save_results(self, result: ExtractionResult, output_path: str):
        """Save results to JSON."""
        import json
        from pathlib import Path
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        print(f"✓ Results saved to {output_path}")


if __name__ == "__main__":
    print("CTI Pipeline Framework - Ready!")
    print("Import and use Pipeline class to build extraction pipelines.")
