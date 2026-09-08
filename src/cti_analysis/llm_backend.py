"""LLM backend for CTI pipeline — llama.cpp server.

Provides generation and embedding via llama-server's REST API.
Supports per-request LoRA adapter selection for the two-pass
extraction architecture (NER adapter on Pass 1, RE adapter on Pass 2).

Usage:
    # Auto-start server from config:
    server = LlamaServer.from_config(cfg.backend)
    server.start()
    backend = LlamaCppBackend(url=server.url)
    text = backend.generate(prompt, lora_id=0)
    server.stop()

    # Or connect to an already-running server:
    backend = LlamaCppBackend(url="http://localhost:8080")
    text = backend.generate(prompt, lora_id=0)
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

GEMMA3_CHAT_TEMPLATE = (
    "<start_of_turn>user\n{prompt}<end_of_turn>\n"
    "<start_of_turn>model\n"
)

# Default llama-server binary location (relative to repo root)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SERVER_BIN = _REPO_ROOT / "tools" / "llama.cpp" / "bin" / "llama-server.exe"


class LlamaServer:
    """Manages the lifecycle of a llama-server process.

    Starts llama-server with the specified model and LoRA adapters,
    waits for it to become healthy, and stops it on request or context exit.

    Usage:
        server = LlamaServer(model_path, lora_paths=[ner_path, re_path])
        server.start()
        # ... use LlamaCppBackend(url=server.url) ...
        server.stop()

    Or as a context manager:
        with LlamaServer(model_path, lora_paths=[ner, re]) as server:
            backend = LlamaCppBackend(url=server.url)
    """

    def __init__(
        self,
        model_path: str,
        lora_paths: Optional[List[str]] = None,
        host: str = "127.0.0.1",
        port: int = 8080,
        n_gpu_layers: int = 99,
        ctx_size: int = 4096,
        flash_attn: bool = True,
        server_bin: Optional[str] = None,
    ):
        self.model_path = model_path
        self.lora_paths = lora_paths or []
        self.host = host
        self.port = port
        self.n_gpu_layers = n_gpu_layers
        self.ctx_size = ctx_size
        self.flash_attn = flash_attn
        self.server_bin = server_bin or str(_DEFAULT_SERVER_BIN)
        self._proc: Optional[subprocess.Popen] = None

    @classmethod
    def from_config(cls, backend_cfg) -> "LlamaServer":
        """Create a LlamaServer from a BackendConfig."""
        lora_paths = []
        if backend_cfg.ner_lora_path:
            lora_paths.append(backend_cfg.ner_lora_path)
        if backend_cfg.re_lora_path:
            lora_paths.append(backend_cfg.re_lora_path)

        # Parse port from URL; always bind to 0.0.0.0 so both IPv4 and IPv6 work
        url = backend_cfg.url or "http://localhost:8080"
        host = "0.0.0.0"
        port = 8080
        if ":" in url.split("//")[-1]:
            parts = url.split("//")[-1].split(":")
            port = int(parts[-1].rstrip("/"))

        return cls(
            model_path=backend_cfg.base_model_path,
            lora_paths=lora_paths,
            host=host,
            port=port,
        )

    @property
    def url(self) -> str:
        """Connectable URL (always localhost, not the bind address)."""
        return f"http://localhost:{self.port}"

    def start(self, timeout: int = 120) -> None:
        """Start llama-server and wait for it to become healthy."""
        if self._proc is not None and self._proc.poll() is None:
            logger.info("llama-server already running (pid=%d)", self._proc.pid)
            return

        cmd = [
            self.server_bin,
            "--model", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--n-gpu-layers", str(self.n_gpu_layers),
            "--ctx-size", str(self.ctx_size),
        ]
        if self.flash_attn:
            cmd += ["--flash-attn", "on"]
        for lora in self.lora_paths:
            cmd += ["--lora", lora]

        logger.info("Starting llama-server: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )

        # Wait for healthy
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = requests.get(f"{self.url}/health", timeout=10)
                if resp.status_code == 200:
                    logger.info("llama-server healthy (pid=%d)", self._proc.pid)
                    return
            except (requests.ConnectionError, requests.ReadTimeout):
                pass
            # Check if process died
            if self._proc.poll() is not None:
                out = self._proc.stdout.read().decode(errors="replace") if self._proc.stdout else ""
                raise RuntimeError(f"llama-server exited with code {self._proc.returncode}:\n{out[:2000]}")
            time.sleep(2)

        self.stop()
        raise TimeoutError(f"llama-server did not become healthy within {timeout}s")

    def stop(self) -> None:
        """Stop the llama-server process."""
        if self._proc is None:
            return
        if self._proc.poll() is not None:
            self._proc = None
            return
        logger.info("Stopping llama-server (pid=%d)...", self._proc.pid)
        try:
            self._proc.terminate()
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=5)
        self._proc = None

    def __enter__(self) -> "LlamaServer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


class LlamaCppBackend:
    """Client for llama-server's native REST API."""

    def __init__(
        self,
        url: str = "http://localhost:8080",
        timeout: int = 300,
        chat_template: Optional[str] = None,
    ):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.chat_template = chat_template or GEMMA3_CHAT_TEMPLATE
        self._session = requests.Session()

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout: Optional[int] = None,
        lora_id: Optional[int] = None,
    ) -> str:
        """Generate a completion from llama-server.

        Args:
            prompt: Raw prompt text (will be wrapped in chat template).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout (overrides default).
            lora_id: LoRA adapter index to use (0-based load order).
                     None = no adapter selection (server default).
        """
        formatted = self.chat_template.format(prompt=prompt)

        payload = {
            "prompt": formatted,
            "temperature": temperature,
            "n_predict": max_tokens,
            "stream": False,
            "cache_prompt": False,
        }

        if lora_id is not None:
            payload["lora"] = [{"id": lora_id, "scale": 1.0}]

        req_timeout = timeout or self.timeout

        for attempt in range(3):
            try:
                resp = self._session.post(
                    f"{self.url}/completion",
                    json=payload,
                    timeout=req_timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data.get("content", "")
                # Strip any trailing end-of-turn tokens
                content = content.replace("<end_of_turn>", "").strip()
                return content
            except requests.exceptions.Timeout:
                logger.warning(
                    "llama-server timeout (attempt %d/3)", attempt + 1
                )
                time.sleep(3)
            except requests.exceptions.ConnectionError:
                logger.warning(
                    "llama-server connection error (attempt %d/3)", attempt + 1
                )
                time.sleep(5)
            except Exception as e:
                logger.warning(
                    "llama-server error (attempt %d/3): %s", attempt + 1, e
                )
                time.sleep(3)

        logger.error("llama-server failed after 3 attempts")
        return ""

    def embed(self, text: str) -> List[float]:
        """Get embeddings from llama-server's /embedding endpoint.

        Requires llama-server started with --embedding flag.
        """
        try:
            resp = self._session.post(
                f"{self.url}/embedding",
                json={"content": text},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json().get("embedding", [])
        except Exception as e:
            logger.error("Embedding request failed: %s", e)
            return []

    def health(self) -> bool:
        """Check if llama-server is healthy."""
        try:
            resp = self._session.get(
                f"{self.url}/health", timeout=5
            )
            return resp.status_code == 200
        except Exception:
            return False
