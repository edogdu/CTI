import argparse
import json
import subprocess
from pathlib import Path

CONFIG_PATH = Path("config/run_config.json")

def update_run_config(args):
    #Update run_config.json based on command-line arguments.
    config = {
        "modules": {
            "consensus": args.consensus,
            "semantic_chunking": args.semantic_chunking,
            "validation": args.validation,
            "reranking": args.reranking,
        }
    }

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)

def run_pipeline():
    #Execute the main pipeline
    print("[Run] Starting ablation test pipeline...\n")
    subprocess.run(["python", "main.py"], check=True)

def parse_args():
    #Parse command-line arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--consensus", action="store_true", help="Enable consensus module")
    parser.add_argument("--semantic_chunking", action="store_true", help="Enable semantic chunking module")
    parser.add_argument("--validation", action="store_true", help="Enable validation module")
    parser.add_argument("--reranking", action="store_true", help="Enable cross-encoder reranking module")

    return parser.parse_args()

def main():
    args = parse_args()
    update_run_config(args)
    run_pipeline()

if __name__ == "__main__":
    main()
