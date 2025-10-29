# scripts/get_foundation_sec.py
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

VARIANTS = {
    "base": "fdtn-ai/Foundation-Sec-8B",
    "instruct": "fdtn-ai/Foundation-Sec-8B-Instruct",
    "gguf-q4": "fdtn-ai/Foundation-Sec-8B-Q4_K_M-GGUF",
    "gguf-q8": "fdtn-ai/Foundation-Sec-8B-Q8_0-GGUF",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, default="base")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    repo_id = VARIANTS[args.variant]
    dest = Path(args.out) / repo_id.replace("/", "__")
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo_id} -> {dest}")
    snapshot_download(repo_id=repo_id, local_dir=str(dest), local_dir_use_symlinks=False)
    print("Done.")

if __name__ == "__main__":
    main()
