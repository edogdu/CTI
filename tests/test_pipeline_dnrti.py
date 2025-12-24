import json
from pathlib import Path

from cti_analysis import pipeline
from cti_analysis.models.documents import RawDocument


def test_pipeline_dnrti(tmp_path, monkeypatch):
    # Use the real DNRTI dataset file and preserve metadata
    dnrti_path = Path(r"C:\CTI\datasets\DNRTI\dnrti_aug_stix2_je copy.json")
    assert dnrti_path.exists(), "DNRTI dataset file is missing"

    import json

    def _mock_loader(cfg):
        data = json.loads(dnrti_path.read_text(encoding="utf-8"))
        docs = []
        for i, entry in enumerate(data):
            docs.append(
                RawDocument(
                    doc_id=f"dnrti_{i}",
                    source_path=str(dnrti_path),
                    text=entry.get("text", ""),
                    meta={
                        "entities": entry.get("entities") or [],
                        "relations": entry.get("relations") or [],
                        "index": i,
                    },
                )
            )
        return docs

    # Monkeypatch to ensure consistent loader for the test
    monkeypatch.setattr(pipeline, "load_raw_documents", _mock_loader)

    # Act
    exit_code = pipeline.run_pipeline()

    # Assert
    assert exit_code == 0
    results = list(Path("results").rglob("*.json*"))
    assert results, "Expected some results written under project results/"

