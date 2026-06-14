import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import google.generativeai as genai
from groq import Groq
from google.api_core.exceptions import GoogleAPIError, ResourceExhausted

from backend.core.config import settings
from backend.core.logging_config import log_event

_gemini_configured = False
_GEMINI_TIMEOUT = 90.0
_gemini_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gemini-llm")


def _configure_gemini() -> None:
    global _gemini_configured
    if _gemini_configured:
        return
    if settings.gemini_api_key:
        genai.configure(api_key=settings.gemini_api_key)
    _gemini_configured = True


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


_GEMINI_MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
]

_discovered_models: list[str] | None = None


def _get_gemini_model_candidates() -> list[str]:
    """Discover available Gemini models once, with safe static fallbacks."""
    global _discovered_models
    if _discovered_models is not None:
        return _discovered_models

    _configure_gemini()
    discovered: list[str] = []
    try:
        for model in genai.list_models():
            methods = getattr(model, "supported_generation_methods", [])
            if "generateContent" not in methods:
                continue
            name = model.name.replace("models/", "")
            if "gemini" in name.lower() and "embedding" not in name.lower():
                discovered.append(name)
    except Exception:
        _discovered_models = list(_GEMINI_MODEL_CANDIDATES)
        return _discovered_models

    priority = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.5-pro",
        "gemini-3-flash-preview",
    ]
    ordered = [name for name in priority if name in discovered]
    ordered.extend(name for name in discovered if name not in ordered)
    _discovered_models = ordered or list(_GEMINI_MODEL_CANDIDATES)
    return _discovered_models


def _call_gemini(prompt: str, system_instruction: str, temperature: float, model_name: str | None = None) -> str:
    _configure_gemini()
    
    candidates = _get_gemini_model_candidates()
    if model_name:
        normalized = model_name.replace("models/", "")
        models = [normalized] + [m for m in candidates if m != normalized]
    else:
        models = candidates

    last_error = None
    for m in models:
        try:
            model = genai.GenerativeModel(model_name=m, system_instruction=system_instruction)
            response = model.generate_content(
                prompt,
                generation_config={"temperature": temperature},
            )
            return getattr(response, "text", "") or ""
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"All Gemini models failed. Last error: {last_error}") from last_error



def _call_groq(prompt: str, system_instruction: str, temperature: float, timeout: float = 60.0) -> str:
    client = Groq(api_key=settings.groq_api_key, timeout=timeout)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""


def call_llm(
    prompt: str,
    system_instruction: str,
    temperature: float = 0.3,
    use_groq: bool = False,
    agent: str = "SYSTEM",
    user_id: str | None = None,
    analysis_id: str | None = None,
    timeout: float = _GEMINI_TIMEOUT,
    model_name: str | None = None,
) -> str:
    start = time.perf_counter()
    model_used = model_name or ("groq" if use_groq else (_get_gemini_model_candidates()[0] if not use_groq else "groq"))

    try:
        if use_groq:
            result = _call_groq(prompt, system_instruction, temperature, timeout=timeout)
        else:
            future = _gemini_executor.submit(_call_gemini, prompt, system_instruction, temperature, model_name)
            try:
                result = future.result(timeout=timeout)
            except TimeoutError:
                future.cancel()
                raise TimeoutError(f"Gemini LLM call timed out after {timeout}s")
        duration_ms = int((time.perf_counter() - start) * 1000)
        log_event(
            agent=agent,
            user_id=user_id,
            analysis_id=analysis_id,
            event="llm_call",
            duration_ms=duration_ms,
            details={
                "model": model_used,
                "token_estimate": _estimate_tokens(prompt + system_instruction),
            },
        )
        return result
    except Exception as exc:
        duration_ms = int((time.perf_counter() - start) * 1000)
        log_event(
            level=40,
            agent=agent,
            user_id=user_id,
            analysis_id=analysis_id,
            event="llm_fallback",
            duration_ms=duration_ms,
            details={"model": model_used, "error_type": type(exc).__name__},
            exc_info=True,
        )
        if use_groq:
            raise

        try:
            time.sleep(2)
            fallback_start = time.perf_counter()
            result = _call_groq(prompt, system_instruction, temperature, timeout=timeout)
            fallback_duration = int((time.perf_counter() - fallback_start) * 1000)
            log_event(
                agent=agent,
                user_id=user_id,
                analysis_id=analysis_id,
                event="llm_call",
                duration_ms=fallback_duration,
                details={
                    "model": "groq-llama-3.3-70b-versatile",
                    "token_estimate": _estimate_tokens(prompt + system_instruction),
                },
            )
            return result
        except Exception:
            raise


def _strip_json_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "", 1).replace("```", "", 1).strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    return cleaned


def _extract_json_object(text: str) -> str:
    """Best-effort extraction when the model wraps JSON in prose."""
    cleaned = _strip_json_fences(text)
    if cleaned.startswith("{") or cleaned.startswith("["):
        return cleaned

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def call_llm_json(
    prompt: str,
    system_instruction: str,
    schema_hint: str,
    temperature: float = 0.3,
    use_groq: bool = False,
    agent: str = "SYSTEM",
    user_id: str | None = None,
    analysis_id: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    response_text = call_llm(
        prompt=prompt,
        system_instruction=system_instruction,
        temperature=temperature,
        use_groq=use_groq,
        agent=agent,
        user_id=user_id,
        analysis_id=analysis_id,
        model_name=model_name,
    )
    cleaned = _extract_json_object(response_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        retry_prompt = f"{prompt}\n\nReturn ONLY valid JSON. Schema hint: {schema_hint}"
        response_text = call_llm(
            prompt=retry_prompt,
            system_instruction=system_instruction,
            temperature=0.0,
            use_groq=use_groq,
            agent=agent,
            user_id=user_id,
            analysis_id=analysis_id,
            model_name=model_name,
        )
        cleaned = _extract_json_object(response_text)
        return json.loads(cleaned)

