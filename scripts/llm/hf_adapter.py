from typing import List
import os, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, BitsAndBytesConfig

# detect bitsandbytes availability
try:
    import bitsandbytes as _bnb  # noqa: F401
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

class HFLLM:
    def __init__(self, model_id_or_dir: str, max_new_tokens: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir)

        # ---- choose load strategy ----
        # options: auto | cpu | 8bit | 4bit
        strategy = os.getenv("HF_LOAD_STRATEGY", "cpu").lower()
        model_kwargs = {}

        if strategy == "4bit":
            if _HAS_BNB:
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs.update(quantization_config=bnb_cfg, device_map="auto")
            else:
                print("[HFLLM] bitsandbytes not found; falling back to CPU")
                model_kwargs.update(device_map={"": "cpu"}, torch_dtype=torch.float32, low_cpu_mem_usage=True)

        elif strategy == "8bit":
            if _HAS_BNB:
                bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
                model_kwargs.update(quantization_config=bnb_cfg, device_map="auto")
            else:
                print("[HFLLM] bitsandbytes not found; falling back to CPU")
                model_kwargs.update(device_map={"": "cpu"}, torch_dtype=torch.float32, low_cpu_mem_usage=True)

        elif strategy == "auto" and torch.cuda.is_available():
            model_kwargs.update(device_map="auto", torch_dtype=torch.float16)
        else:
            # SAFE DEFAULT: run fully on CPU to avoid disk-offload errors
            model_kwargs.update(device_map={"": "cpu"}, torch_dtype=torch.float32, low_cpu_mem_usage=True)

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id_or_dir, **model_kwargs)
        except ValueError:
            # If auto dispatch fails (e.g., no GPU), fall back cleanly to CPU
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id_or_dir,
                device_map={"": "cpu"},
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )

        # set pad token once to avoid the runtime warning
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        self.max_new_tokens = max_new_tokens

        # --- embedding model (keep on CPU to save VRAM) ---
        emb_id = os.getenv("HF_EMBED_MODEL", "nomic-ai/nomic-embed-text-v1")
        self.embed_tokenizer = AutoTokenizer.from_pretrained(emb_id)
        self.embed_model = AutoModel.from_pretrained(
            emb_id, trust_remote_code=True, device_map={"": "cpu"}
        )
        self.embed_model.eval()

        print(f"[HFLLM] strategy={strategy} | bnb={_HAS_BNB} | device_map={getattr(self.model, 'hf_device_map', None)}")

    def invoke(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
            )
        full = self.tokenizer.decode(out[0], skip_special_tokens=True)
        given = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        return full[len(given):].strip()

    def embed(self, texts: List[str]) -> List[List[float]]:
        import numpy as np
        vecs: List[List[float]] = []
        device = next(self.embed_model.parameters()).device
        for t in texts:
            tok = self.embed_tokenizer(t, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                out = self.embed_model(**tok)
                h = out.last_hidden_state.mean(dim=1)  # mean pool
                v = torch.nn.functional.normalize(h, p=2, dim=1).squeeze(0).cpu().numpy().tolist()
                vecs.append(v)
        return vecs