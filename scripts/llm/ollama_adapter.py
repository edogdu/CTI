# llm/ollama_adapter.py
import os, json, requests
from typing import List

class OllamaLLM:
    def __init__(self, model: str, base_url: str = "http://localhost:11434", num_ctx: int = 4096):
        self.model = model
        self.base = base_url.rstrip("/")
        self.num_ctx = num_ctx

    # Text generation
    def invoke(self, prompt: str) -> str:
        url = f"{self.base}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {"num_ctx": self.num_ctx, "temperature": 0.2, "top_p": 0.9},
            "stream": False
        }
        r = requests.post(url, json=payload, timeout=600)
        r.raise_for_status()
        out = r.json().get("response","")
        return out.strip()

    # Embeddings
    def embed(self, texts: List[str]) -> List[List[float]]:
        url = f"{self.base}/api/embeddings"
        vecs = []
        for t in texts:
            r = requests.post(url, json={"model": os.getenv("EMBED_MODEL", "nomic-embed-text"), "prompt": t}, timeout=120)
            r.raise_for_status()
            v = r.json().get("embedding", [])
            vecs.append(v)
        return vecs
