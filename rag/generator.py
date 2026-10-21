"""
CampusIQ Gemini Generator
Handles LLM generation via the official google-genai SDK.
Implements:
1. Secure credential handling (.env only, no hardcoded secrets).
2. Exponential backoff retry for free-tier rate limits (15 RPM / 429 RESOURCE_EXHAUSTED).
3. Local disk caching for deterministic, reproducible, and quota-friendly benchmarks.
4. Structured token usage and latency tracking.
"""

import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, Optional
from dotenv import load_dotenv

logger = logging.getLogger("campusiq.rag.generator")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CACHE_FILE = os.path.join(BASE_DIR, "eval", "results", "rag_cache.json")


class GeminiGenerator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        cache_file: str = DEFAULT_CACHE_FILE,
        use_cache: bool = True,
        client: Optional[Any] = None,
        max_retries: int = 5,
        base_backoff_sec: float = 2.0,
    ):
        load_dotenv(os.path.join(BASE_DIR, ".env"))

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self.cache_file = cache_file
        self.use_cache = use_cache
        self.max_retries = max_retries
        self.base_backoff_sec = base_backoff_sec
        self.fallback_models = ["gemini-3.5-flash-lite", "gemini-flash-lite-latest"]

        self._cache: Dict[str, Dict[str, Any]] = {}
        if self.use_cache:
            self._load_cache()

        if client is not None:
            self.client = client
        else:
            if not self.api_key:
                raise ValueError(
                    "GEMINI_API_KEY is not set. Please add it to your .env file or environment variables."
                )
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "google-genai package is not installed. Run 'pip install google-genai' to install."
                )

    def _load_cache(self) -> None:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.debug(f"Loaded {len(self._cache)} cached generations from {self.cache_file}")
            except Exception as e:
                logger.warning(f"Could not read cache file {self.cache_file}: {e}")
                self._cache = {}

    def _save_cache(self) -> None:
        if not self.use_cache:
            return
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Could not persist cache to {self.cache_file}: {e}")

    def _cache_key(self, system_instruction: str, contents: str) -> str:
        payload = f"{system_instruction}::{contents}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Generates text using Gemini with backoff retry, model fallbacks, and caching.
        """
        sys_inst = system_instruction or ""
        ckey = self._cache_key(sys_inst, prompt)

        if self.use_cache and ckey in self._cache:
            cached_data = dict(self._cache[ckey])
            cached_data["cached"] = True
            cached_data["latency_ms"] = 0.0
            return cached_data

        start_time = time.perf_counter()
        attempt = 0
        last_error = None
        current_model = self.model

        while attempt < self.max_retries:
            try:
                config: Dict[str, Any] = {"temperature": temperature}
                if sys_inst:
                    config["system_instruction"] = sys_inst

                response = self.client.models.generate_content(
                    model=current_model,
                    contents=prompt,
                    config=config,
                )

                elapsed_ms = (time.perf_counter() - start_time) * 1000

                text = response.text.strip() if response and response.text else ""
                usage = getattr(response, "usage_metadata", None)
                in_tok = getattr(usage, "prompt_token_count", 0) if usage else 0
                out_tok = getattr(usage, "candidates_token_count", 0) if usage else 0
                total_tok = getattr(usage, "total_token_count", in_tok + out_tok) if usage else in_tok + out_tok

                result = {
                    "text": text,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "total_tokens": total_tok,
                    "latency_ms": round(elapsed_ms, 2),
                    "cached": False,
                    "model": current_model,
                }

                if self.use_cache:
                    self._cache[ckey] = result
                    self._save_cache()

                return result

            except Exception as e:
                attempt += 1
                last_error = e
                err_str = str(e)
                is_rate_limit = (
                    "429" in err_str
                    or "RESOURCE_EXHAUSTED" in err_str
                    or "rate limit" in err_str.lower()
                    or "503" in err_str
                    or "UNAVAILABLE" in err_str
                )

                if is_rate_limit and attempt < self.max_retries:
                    # Switch to fallback model on persistent 503
                    if ("503" in err_str or "UNAVAILABLE" in err_str) and attempt >= 2:
                        fallback_idx = (attempt - 2) % len(self.fallback_models)
                        current_model = self.fallback_models[fallback_idx]
                        logger.info(f"Switching to fallback model '{current_model}' due to 503 on primary.")

                    backoff = min(30.0, self.base_backoff_sec * (1.8 ** (attempt - 1)))
                    logger.warning(
                        f"Gemini API retryable error (attempt {attempt}/{self.max_retries}). "
                        f"Retrying in {backoff:.1f}s. Error: {err_str[:120]}"
                    )
                    time.sleep(backoff)
                else:
                    logger.error(f"Generation failed on attempt {attempt}: {err_str}")
                    break

        raise RuntimeError(
            f"Failed to generate content after {attempt} attempts: {last_error}"
        ) from last_error

