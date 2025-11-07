# extractors/semantic_extractor_adapter.py
from core.pipeline import Extractor
from core.data_models import ExtractionResult, Triple
from core.config import Config
from core.normalization import normalize_name
# import your semantic function
from triple_extraction.extraction_semantic_only_v1 import extract_triples_from_text

class SemanticExtractorAdapter(Extractor):
    def __init__(self, config: Config = None):
        super().__init__("semantic_extractor", config)

    def extract(self, text: str) -> ExtractionResult:
        res = ExtractionResult()
        tuples = extract_triples_from_text(
            text,
            model_name=self.config.MODEL_NAME,
            ollama_base_url=self.config.OLLAMA_BASE_URL
        )
        # tuples are (s,p,o); wrap as Triples with unknown types if unavailable
        for s, p, o in tuples:
            res.triples.append(Triple(
                subject=normalize_name(s),
                subject_type="unknown",
                predicate=p,
                object=normalize_name(o),
                object_type="unknown",
                confidence=0.95,
                evidence=text[:200]
            ))
        res.valid_triples = list(res.triples)
        return res
