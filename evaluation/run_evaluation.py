# run_evaluation.py

import json
from pathlib import Path
import argparse

# Your project imports
from core.config import Config
from core.pipeline import Pipeline
from evaluation.evaluator import Evaluator

# Optional if you created these helpers. If not, remove these two imports and their usage.
try:
    from core.normalization import to_triple_tuple, to_unified_dict  # noqa: F401
except Exception:
    to_triple_tuple = None
    to_unified_dict = None

# Optional registry for switching extractors
try:
    from extractors.registry import build_extractor
except Exception:
    build_extractor = None


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=str, choices=["cti-hal", "anno-ctr", "dnrti", "all"], default="all")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--threshold", type=float, default=0.66)
    parser.add_argument("--config", type=str)

    # model switch
    parser.add_argument("--model", type=str, default=None,
                        help="Model name string for your backend, e.g. 'gemma-2-9b'")

    # dataset paths
    parser.add_argument("--cti-hal-path", type=str, default="./data/CTI-HAL")
    parser.add_argument("--anno-ctr-path", type=str, default="./data/ANNO-CTR")
    parser.add_argument("--dnrti-path", type=str, default="./data/DNRTI")

    # pipeline toggles
    parser.add_argument("--extractors", type=str, default="multi-agent",
                        help="Comma-separated: multi-agent,semantic")
    parser.add_argument("--enable-stix", action="store_true")
    parser.add_argument("--enable-markov", action="store_true")

    # output
    parser.add_argument("--output", type=str, default="output/summary.json")
    parser.add_argument("--jsonl", type=str, default="output/eval_records.jsonl")
    parser.add_argument("--tag", type=str, default="")


def build_pipeline(config: Config, extractor_names: list[str]) -> Pipeline:
    pipe = Pipeline(config)

    # extractors
    if build_extractor is not None:
        for name in extractor_names:
            pipe.add_extractor(build_extractor(name, config))
    else:
        # fallback: always add multi-agent if registry missing
        from extractors.multi_agent_extractor import MultiAgentExtractor
        pipe.add_extractor(MultiAgentExtractor(config))

    # optional consensus filter
    try:
        from consensus import consensus_filter
        pipe.add_filter(consensus_filter(config))
    except Exception:
        print("Note: Consensus filter not available")

    # optional markov repairer (guarded by flag in main to match args.enable_markov)
    # we add here but only if asked by caller; so leave import for main()

    # optional STIX validator (same note as above)

    return pipe


def main():
    parser = argparse.ArgumentParser(description="CTI Pipeline Evaluation")
    add_cli_args(parser)
    args = parser.parse_args()

    # Load config
    config = Config(args.config) if args.config else Config()

    # Allow switching the model at runtime
    if args.model:
        # these attribute names follow your Config style; adjust if yours differ
        config.MODEL_NAME = args.model

    # Build pipeline with chosen extractors
    extractor_names = [s.strip() for s in args.extractors.split(",") if s.strip()]
    pipeline = build_pipeline(config, extractor_names)

    # Attach optional stages based on flags
    if args.enable_markov:
        try:
            from repair.markov_repair import MarkovRepairer
            pipeline.add_repairer(MarkovRepairer(config))
        except Exception as e:
            print(f"Note: Markov repairer not available: {e}")

    if args.enable_stix:
        try:
            from validators.stix_validator import STIXValidator
            pipeline.add_validator(STIXValidator(config))
        except Exception as e:
            print(f"Note: STIX validator not available: {e}")

    # Evaluator
    evaluator = Evaluator(pipeline, config)

    # JSONL sink for per-sample
    records_path = Path(args.jsonl)
    records_path.parent.mkdir(parents=True, exist_ok=True)

    def write_record(rec: dict) -> None:
        with open(records_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Helper to evaluate one dataset and write per-sample rows if provided
    def eval_one(ds_name: str, ds_path: str):
        result = evaluator.evaluate_dataset(
            ds_name,
            ds_path,
            num_samples=args.samples,
            split=args.split,
            threshold=args.threshold,
            debug_first=getattr(config, "DEBUG_FIRST", 2)
        )
        for rec in result.get("per_sample", []):
            rec.setdefault("dataset", ds_name)
            rec.setdefault("extractors", extractor_names)
            rec.setdefault("tag", args.tag)
            write_record(rec)
        return result

    results = {}

    if args.dataset == "all":
        results["cti-hal"] = eval_one("cti-hal", args.cti_hal_path)
        results["anno-ctr"] = eval_one("anno-ctr", args.anno_ctr_path)
        results["dnrti"]   = eval_one("dnrti",   args.dnrti_path)
    elif args.dataset == "cti-hal":
        results["cti-hal"] = eval_one("cti-hal", args.cti_hal_path)
    elif args.dataset == "anno-ctr":
        results["anno-ctr"] = eval_one("anno-ctr", args.anno_ctr_path)
    elif args.dataset == "dnrti":
        results["dnrti"] = eval_one("dnrti", args.dnrti_path)
    else:
        raise ValueError("Unknown dataset selection")

    # Write summary JSON
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "tag": args.tag,
        "extractors": extractor_names,
        "samples_per_dataset": args.samples,
        "threshold": args.threshold,
        "model": getattr(config, "MODEL_NAME", None),
        "results": results
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Results saved to {args.output}")
    print(f"✓ Per-sample JSONL at {args.jsonl}")


if __name__ == "__main__":
    main()
