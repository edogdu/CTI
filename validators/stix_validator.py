# validators/stix_validator.py
from typing import List
from core.pipeline import Validator
from core.data_models import Triple, ExtractionResult
from core.stix_rules import det_fix, validate

class STIXValidator(Validator):
    def __init__(self, config=None):
        super().__init__("stix_validator")
        self.config = config

    def process(self, result: ExtractionResult) -> ExtractionResult:
        valid: List[Triple] = []
        errors: List[str] = []
        for t in result.triples:
            # normalize then check
            t_norm = det_fix({
                "subject": {"name": t.subject, "type": t.subject_type},
                "predicate": t.predicate,
                "object": {"name": t.object, "type": t.object_type},
            })
            ok, reasons = validate(t_norm)
            if ok:
                # push back into Triple
                t.subject = t_norm["subject"]["name"]
                t.subject_type = t_norm["subject"]["type"]
                t.predicate = t_norm["predicate"]
                t.object = t_norm["object"]["name"]
                t.object_type = t_norm["object"]["type"]
                valid.append(t)
            else:
                errors.append(f"{t.subject} {t.predicate} {t.object} :: {reasons}")
        result.valid_triples = valid
        result.errors = (result.errors or []) + errors
        return result
