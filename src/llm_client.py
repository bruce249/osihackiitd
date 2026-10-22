"""Cached OpenAI JSON chat calls and token accounting.

The API key is read only from the environment variable OPENAI_API_KEY.
It is never written to disk, included in the cache, or logged.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USAGE_PATH = ROOT / "results" / "llm_usage.json"
DEFAULT_MODEL = "gpt-6-luna"

# Counts HTTP attempts. Cache hits do not increment this.
live_calls = 0

_thread_locks_guard = threading.Lock()
_thread_locks: dict[str, threading.Lock] = {}
_params_lock = threading.Lock()
# Temperature is tried once. If the API rejects it, later calls omit it.
_send_temperature = True
# reasoning_effort is omitted: there is no evidence gpt-6-luna accepts it.
_send_json_mode = True

_SECRET_RE = re.compile(r"sk-[A-Za-z0-9_\-]{8,}")
_USAGE_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached_tokens",
    "n_calls",
    "n_cache_hits",
)


class LLMCacheMissing(RuntimeError):
    """No cache file exists and OPENAI_API_KEY is unset."""


def redact(text: str) -> str:
    """Remove API-key material from a message before it is logged."""
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        text = text.replace(key, "[redacted]")
    return _SECRET_RE.sub("sk-[redacted]", text)


def chat_json(system: str, user: str, cache_path: str) -> dict:
    """Return the assistant JSON object for this prompt.

    If ``cache_path`` exists, return the cached assistant payload. An envelope
    ``{"request", "response", "usage"}`` is unwrapped to ``response``.
    If the cache is missing and OPENAI_API_KEY is unset, raise RuntimeError.
    A live call retries three times with exponential backoff, then stores the
    envelope. Cache hits increment ``n_cache_hits`` only.
    """
    path = _abs_path(cache_path)
    with file_lock(str(path)):
        if path.is_file():
            payload = _load_cached_response(path)
            _record_usage(cache_hit=True)
            return payload
        if not _api_key_set():
            raise LLMCacheMissing(
                f"No cached LLM response at {path} and OPENAI_API_KEY is not set. "
                "Set the key or provide a cache file."
            )
        model = os.environ.get("LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        response, usage, request = _call_with_retries(model, system, user)
        _atomic_write_json(
            path,
            {"request": request, "response": response, "usage": usage},
        )
        _record_usage(cache_hit=False, usage=usage)
        return response


def append_jsonl(path, record: dict) -> None:
    """Append one JSON object as a line, under an exclusive lock."""
    target = _abs_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with file_lock(str(target)):
        with target.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.flush()


@contextmanager
def file_lock(path: str, timeout: float = 60.0):
    """Lock ``path`` for threads and processes.

    Uses portalocker when it is installed. Otherwise a threading.Lock plus an
    exclusive lock file created with os.open(O_CREAT | O_EXCL).
    """
    path = os.path.abspath(str(path))
    thread_lock = _lock_for(path)
    if not thread_lock.acquire(timeout=timeout):
        raise TimeoutError(f"Timed out waiting for thread lock on {path}")
    try:
        lock_path = path + ".lock"
        Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
        portal_cm = _portalocker_lock(lock_path, timeout)
        if portal_cm is not None:
            with portal_cm:
                yield
        else:
            fd = _acquire_excl(lock_path, timeout)
            try:
                yield
            finally:
                _release_excl(fd, lock_path)
    finally:
        thread_lock.release()


def _portalocker_lock(lock_path: str, timeout: float):
    try:
        import portalocker
    except ImportError:
        return None
    try:
        return portalocker.Lock(lock_path, mode="a", timeout=timeout)
    except TypeError:
        return None


def _lock_for(path: str) -> threading.Lock:
    with _thread_locks_guard:
        lock = _thread_locks.get(path)
        if lock is None:
            lock = threading.Lock()
            _thread_locks[path] = lock
        return lock


def _acquire_excl(lock_path: str, timeout: float) -> int:
    deadline = time.monotonic() + timeout
    stale_seconds = 120.0
    while True:
        try:
            return os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for lock {lock_path}")
            try:
                age = time.time() - os.path.getmtime(lock_path)
            except OSError:
                age = 0.0
            if age > stale_seconds:
                try:
                    os.remove(lock_path)
                    continue
                except OSError:
                    pass
            time.sleep(0.02)


def _release_excl(fd: int, lock_path: str) -> None:
    try:
        os.close(fd)
    except OSError:
        pass
    for _ in range(10):
        try:
            os.remove(lock_path)
            return
        except FileNotFoundError:
            return
        except OSError:
            time.sleep(0.01)


def _abs_path(path) -> Path:
    target = Path(path)
    if not target.is_absolute():
        target = ROOT / target
    return Path(os.path.abspath(target))


def _api_key_set() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def _load_cached_response(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Cache file is not valid JSON: {path}") from exc
    if isinstance(data, dict) and "response" in data and (
        "request" in data or "usage" in data
    ):
        data = data["response"]
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Cached response string is not valid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Cached LLM response is not a JSON object: {path}")
    return data


def _empty_usage() -> dict:
    return {key: 0 for key in _USAGE_KEYS}


def _record_usage(*, cache_hit: bool, usage: dict | None = None) -> None:
    path = USAGE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(str(path)):
        data = _empty_usage()
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                loaded = {}
            if isinstance(loaded, dict):
                data.update(loaded)
        for bad in ("api_key", "openai_api_key", "authorization"):
            data.pop(bad, None)
        for key in _USAGE_KEYS:
            try:
                data[key] = int(data.get(key) or 0)
            except (TypeError, ValueError):
                data[key] = 0
        if cache_hit:
            data["n_cache_hits"] += 1
        else:
            data["n_calls"] += 1
            extra = usage or {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens"):
                try:
                    data[key] += int(extra.get(key) or 0)
                except (TypeError, ValueError):
                    pass
        _atomic_write_json(path, data)


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _call_with_retries(model: str, system: str, user: str) -> tuple[dict, dict, dict]:
    global live_calls
    try:
        from openai import OpenAI
    except ModuleNotFoundError:
        raise RuntimeError(
            "The openai package is not installed. Install it to make live calls, "
            "or provide a cache file."
        ) from None

    client = OpenAI(timeout=60.0, max_retries=0)
    attempt = 0
    param_fixes = 0
    last_error: Exception | None = None
    while attempt < 3:
        use_temp, use_json = _param_flags()
        kwargs: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if use_json:
            kwargs["response_format"] = {"type": "json_object"}
        if use_temp:
            kwargs["temperature"] = 0
        try:
            live_calls += 1
            completion = client.chat.completions.create(**kwargs)
            payload = _loads_json_object(_message_text(completion))
            usage = _usage_from_completion(completion)
            request = {
                "model": model,
                "messages": kwargs["messages"],
            }
            if use_json:
                request["response_format"] = {"type": "json_object"}
            if use_temp:
                request["temperature"] = 0
            return payload, usage, request
        except Exception as exc:
            if isinstance(exc, AssertionError):
                raise
            last_error = exc
            text = redact(str(exc)).lower()
            if use_temp and "temperature" in text and param_fixes < 3:
                _disable_temperature()
                param_fixes += 1
                continue
            if use_json and _json_mode_rejected(text) and param_fixes < 3:
                _disable_json_mode()
                param_fixes += 1
                continue
            attempt += 1
            if attempt >= 3:
                break
            time.sleep(2 ** (attempt - 1))
    raise RuntimeError(redact(f"LLM request failed after retries: {last_error}")) from None


def _param_flags() -> tuple[bool, bool]:
    with _params_lock:
        return _send_temperature, _send_json_mode


def _disable_temperature() -> None:
    global _send_temperature
    with _params_lock:
        _send_temperature = False


def _disable_json_mode() -> None:
    global _send_json_mode
    with _params_lock:
        _send_json_mode = False


def _json_mode_rejected(text: str) -> bool:
    return "response_format" in text or "json_object" in text


def _message_text(completion) -> str:
    try:
        message = completion.choices[0].message
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError("completion has no message content") from exc
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
            else:
                parts.append(str(getattr(part, "text", "") or ""))
        return "".join(parts)
    parsed = getattr(message, "parsed", None)
    if isinstance(parsed, dict):
        return json.dumps(parsed)
    if content is None:
        raise ValueError("empty completion content")
    return str(content)


def _loads_json_object(text: str) -> dict:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, count=1, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("JSON payload is not an object")
    return data


def _usage_from_completion(completion) -> dict:
    usage = getattr(completion, "usage", None)
    raw: dict = {}
    if usage is None:
        raw = {}
    elif hasattr(usage, "model_dump"):
        raw = usage.model_dump()
    elif isinstance(usage, dict):
        raw = usage
    else:
        raw = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None) if details is not None else None
        if cached is not None:
            raw["cached_tokens"] = cached
    prompt = int(raw.get("prompt_tokens") or raw.get("input_tokens") or 0)
    completion_tokens = int(raw.get("completion_tokens") or raw.get("output_tokens") or 0)
    total = int(raw.get("total_tokens") or (prompt + completion_tokens))
    cached_tokens = raw.get("cached_tokens")
    if cached_tokens is None:
        details = raw.get("prompt_tokens_details") or raw.get("input_tokens_details") or {}
        if isinstance(details, dict):
            cached_tokens = details.get("cached_tokens")
    try:
        cached_out = int(cached_tokens or 0)
    except (TypeError, ValueError):
        cached_out = 0
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion_tokens,
        "total_tokens": total,
        "cached_tokens": cached_out,
    }
