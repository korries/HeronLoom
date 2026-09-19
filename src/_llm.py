"""Centralised LLM config resolution and HTTP call utilities."""

from __future__ import annotations

import os
import time

import requests

from utils.logger import get_logger

logger = get_logger(__name__)

# Two request shapes are supported, set per slot via cfg["models"][slot]["api"]:
#   "openai-completions"  → POST {url}, Authorization: Bearer, messages: [system, user, ...]   (default)
#   "anthropic-messages"  → POST {url}, x-api-key + anthropic-version, system: "...", messages: [user, ...]
_DEFAULT_API = "openai-completions"
_ANTHROPIC_VERSION = "2023-06-01"


def get_llm_cfg(cfg: dict, section: str) -> dict:
    """Return the merged LLM config for a pipeline module.

    Merges cfg["models"][slot] (base) with cfg[section] (overrides).
    API key falls back to the slot's env var (<SLOT>_API_KEY, e.g. slot
    "anthropic" → ANTHROPIC_API_KEY) if not set in config.yaml.

    Args:
        cfg: global config loaded from config.yaml.
        section: module section name ("nova", "adept", "naming").

    Returns:
        Ready-to-use config dict.

    Raises:
        ValueError: if the resolved slot is absent from cfg["models"].
    """
    section_cfg = cfg.get(section) or {}
    slot        = section_cfg.get("model_slot", "heavy")
    model_cfg   = cfg.get("models", {}).get(slot, {})

    if not model_cfg:
        raise ValueError(
            f"[LLM] slot '{slot}' not found in cfg['models'] — check config.yaml → models: {slot}:"
        )

    merged = {**model_cfg, **section_cfg}
    merged.setdefault("api", _DEFAULT_API)

    if not merged.get("api_key"):
        merged["api_key"] = os.getenv(f"{slot.upper()}_API_KEY", "")

    return merged


def _headers_for(ncfg: dict) -> dict:
    """Auth headers for the configured api shape."""
    if ncfg.get("api") == "anthropic-messages":
        return {
            "x-api-key":         ncfg.get("api_key", ""),
            "anthropic-version": ncfg.get("anthropic_version", _ANTHROPIC_VERSION),
        }
    return {
        "Authorization": f"Bearer {ncfg.get('api_key', '')}",
        "HTTP-Referer":  "http://localhost",
    }


def _build_payload(ncfg: dict, prompt_system: str, prompt_user: str) -> dict:
    """Request body for the configured api shape.

    ncfg["extra_params"], if set, is merged as-is into the payload. This is
    how any provider-specific field (Ollama's "think", OpenAI's
    "reasoning_effort", Anthropic's "thinking", or anything else) gets sent,
    without this file needing to know about it in advance.
    """
    if ncfg.get("api") == "anthropic-messages":
        # Anthropic: system is a top-level field, not a message; max_tokens
        # is required by the API (default here if config.yaml left it blank).
        payload = {
            "model":      ncfg["model"],
            "stream":     False,
            "max_tokens": ncfg.get("max_tokens") or 4096,
            "system":     prompt_system,
            "messages":   [{"role": "user", "content": prompt_user}],
        }
    else:
        # OpenAI-compatible (OpenAI, OpenRouter, Ollama, vLLM, ...)
        payload = {
            "model":    ncfg["model"],
            "stream":   False,
            "messages": [
                {"role": "system", "content": prompt_system},
                {"role": "user",   "content": prompt_user},
            ],
        }
        # Only sent if explicitly set in config.yaml — leaving a field blank
        # lets the provider apply its own default instead of us guessing one.
        # num_ctx in particular is Ollama-specific; other providers don't expect it.
        if ncfg.get("num_ctx") is not None:
            payload["num_ctx"] = ncfg["num_ctx"]
        if ncfg.get("max_tokens") is not None:
            payload["max_tokens"] = ncfg["max_tokens"]

    # ncfg.get("extra_params", {}) only falls back to {} when the key is
    # ABSENT. When config.yaml has "extra_params:" with nothing after it,
    # the key exists with value None, so .get() returns None here — and
    # payload.update(None) is exactly what crashed in the original traceback.
    payload.update(ncfg.get("extra_params") or {})
    return payload


def _parse_response(ncfg: dict, data: dict) -> str:
    """Extract text from the provider's response shape."""
    if ncfg.get("api") == "anthropic-messages":
        # {"content": [{"type": "text", "text": "..."}, {"type": "thinking", ...}, ...]}
        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        if not text:
            text = "".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking")
        return text.strip()

    if "message" in data:
        message = data["message"]
    else:
        message = data["choices"][0]["message"]
    return (message.get("content") or message.get("reasoning_content") or message.get("reasoning") or "").strip()


def check_llm_available(ncfg: dict) -> bool:
    """Ping the LLM endpoint once; return True if reachable."""
    payload = _build_payload(ncfg, "", "ping")
    payload["max_tokens"] = 1
    try:
        r = requests.post(ncfg["url"], json=payload, timeout=60, headers=_headers_for(ncfg))
        return r.ok
    except Exception:
        return False


def raw_request(ncfg: dict, prompt_system: str, prompt_user: str) -> str:
    """Single HTTP call — no retry. Base brick for llm_call() and custom retry loops.

    Args:
        ncfg: merged LLM config from get_llm_cfg().
        prompt_system: system prompt string.
        prompt_user: user prompt string.

    Returns:
        Raw LLM text (content or reasoning_content).

    Raises:
        requests.exceptions.RequestException: on HTTP failure; caller handles retry.
    """
    payload = _build_payload(ncfg, prompt_system, prompt_user)

    r = requests.post(
        ncfg["url"],
        json=payload,
        timeout=ncfg.get("timeout", 300),
        headers=_headers_for(ncfg),
    )
    if not r.ok:
        logger.warning("HTTP %d — %s", r.status_code, r.text[:300])
    r.raise_for_status()

    return _parse_response(ncfg, r.json())


def llm_call(ncfg: dict, prompt_system: str, prompt_user: str) -> str:
    """OpenAI/Anthropic-compatible HTTP call with exponential-backoff retry.

    Retries the same request until success or max_retries exhausted.

    Args:
        ncfg: merged LLM config from get_llm_cfg().
        prompt_system: system prompt string.
        prompt_user: user prompt string.

    Returns:
        Raw LLM text.

    Raises:
        RuntimeError: if all attempts fail.
    """
    max_retries = ncfg.get("max_retries", 3)

    for attempt in range(1, max_retries + 1):
        try:
            return raw_request(ncfg, prompt_system, prompt_user)
        except Exception as e:
            logger.warning("attempt %d/%d failed: %s: %s", attempt, max_retries, type(e).__name__, e)
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    raise RuntimeError(f"[LLM] all {max_retries} attempts failed")
