"""LLM strategies: llm, llm_anon, and llm_zeroshot.

The campaign calls ``make_strategy()`` with no arguments and records
``strategy.name``. When this process was started with
``--strategy llm_anon`` (or ``llm_zeroshot``) and ``--module`` pointing at
this file, that name is used. Explicit ``mode`` and ``one_shot`` arguments
win, which is how tests and ``recommend()`` select a variant.
"""

from __future__ import annotations

import logging
import sys
import threading

from src.llm_client import LLMCacheMissing, ROOT, append_jsonl, chat_json, redact
from src.prompts import build_messages, parse_picks, row_labels

log = logging.getLogger(__name__)

REASONS_PATH = ROOT / "results" / "llm_reasons.jsonl"

_ALIASES = {
    "llm": ("named", False),
    "llm_anon": ("anon", False),
    "llm_zeroshot": ("named", True),
    "llm_anon_zeroshot": ("anon", True),
}

_warn_lock = threading.Lock()
_warned_no_key = False


def make_strategy(mode=None, one_shot=None, **kwargs):
    mode, one_shot = _resolve_mode(mode, one_shot, kwargs)
    return Strategy(mode=mode, one_shot=one_shot)


class Strategy:
    def __init__(self, mode: str = "named", one_shot: bool = False):
        if mode not in ("named", "anon"):
            raise ValueError("mode must be 'named' or 'anon'")
        self.mode = mode
        self.one_shot = bool(one_shot)
        self.name = _strategy_name(self.mode, self.one_shot)
        self.last_reasons: dict[int, str] = {}

    def propose(self, task, pool_idx, hist_idx, hist_y, q, rng, seed, round_idx) -> list[int]:
        pairs = fetch_picks(
            strategy_name=self.name,
            task=task,
            pool_idx=pool_idx,
            hist_idx=hist_idx,
            hist_y=hist_y,
            k=q,
            seed=seed,
            round_idx=round_idx,
            mode=self.mode,
        )
        self.last_reasons = {idx: reason for idx, reason in pairs}
        return [idx for idx, _reason in pairs]


def fetch_picks(
    strategy_name,
    task,
    pool_idx,
    hist_idx,
    hist_y,
    k,
    seed,
    round_idx,
    mode,
) -> list[tuple[int, str]]:
    """Ask the LLM for up to ``k`` pool rows. Returns ``(row_idx, reason)``."""
    k = int(k)
    pool = _unique_ints(pool_idx)
    if k <= 0 or not pool:
        return []
    pool_set = set(pool)
    path = cache_path_for(strategy_name, task.id, seed, round_idx)
    try:
        system, user = build_messages(task, hist_idx, hist_y, k, mode)
        payload = chat_json(system, user, str(path))
    except LLMCacheMissing:
        _warn_no_key_once()
        return []
    except Exception as exc:
        log.warning(
            "LLM propose failed for %s/%s: %s",
            strategy_name,
            getattr(task, "id", "?"),
            redact(str(exc))[:400],
        )
        return []

    chosen: list[tuple[int, str]] = []
    seen: set[int] = set()
    for idx, reason in parse_picks(payload, task):
        if idx not in pool_set or idx in seen:
            continue
        seen.add(idx)
        chosen.append((int(idx), reason))
        if len(chosen) >= k:
            break
    if chosen:
        try:
            _write_reasons(strategy_name, task, seed, round_idx, chosen)
        except Exception as exc:
            log.warning("could not append LLM reasons: %s", redact(str(exc))[:400])
    return chosen


def cache_path_for(strategy: str, task_id: str, seed, round_idx):
    return (
        ROOT
        / "data"
        / "cache"
        / "llm"
        / str(strategy)
        / str(task_id)
        / str(seed)
        / f"{round_idx}.json"
    )


def _strategy_name(mode: str, one_shot: bool) -> str:
    if mode == "anon" and one_shot:
        return "llm_anon_zeroshot"
    if mode == "anon":
        return "llm_anon"
    if one_shot:
        return "llm_zeroshot"
    return "llm"


def _resolve_mode(mode, one_shot, kwargs) -> tuple[str, bool]:
    hint = kwargs.get("strategy")
    if hint is None:
        hint = kwargs.get("name")
    if isinstance(mode, str) and mode.lower() in _ALIASES:
        hint = mode
        mode = None
    if hint is None and mode is None and one_shot is None:
        hint = _cli_strategy_name()
    if hint is not None and str(hint).lower() in _ALIASES:
        mapped_mode, mapped_shot = _ALIASES[str(hint).lower()]
        if mode is None:
            mode = mapped_mode
        if one_shot is None:
            one_shot = mapped_shot
    if mode is None:
        mode = "named"
    mode_key = str(mode).lower()
    if mode_key not in ("named", "anon"):
        raise ValueError("mode must be 'named' or 'anon'")
    if one_shot is None:
        one_shot = False
    return mode_key, bool(one_shot)


def _cli_strategy_name():
    """Read --strategy when the runner loaded this module from the CLI."""
    argv = sys.argv
    module = _argv_value(argv, "--module")
    strategy = _argv_value(argv, "--strategy")
    if not module or not strategy:
        return None
    norm = module.replace("\\", ".").replace("/", ".")
    if norm.endswith("llm_agent") or norm.endswith("llm_agent.py"):
        return strategy
    return None


def _argv_value(argv, flag: str):
    if flag in argv:
        index = argv.index(flag)
        if index + 1 < len(argv):
            return argv[index + 1]
    prefix = flag + "="
    for arg in argv:
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def _warn_no_key_once() -> None:
    global _warned_no_key
    with _warn_lock:
        if _warned_no_key:
            return
        _warned_no_key = True
    log.warning(
        "LLM cache missing and OPENAI_API_KEY is unset; returning no picks. "
        "Set the key or provide a cache file."
    )


def _write_reasons(strategy_name, task, seed, round_idx, chosen) -> None:
    cols = [str(col) for col in task.var_cols]
    for idx, reason in chosen:
        labels = row_labels(task, idx)
        append_jsonl(
            REASONS_PATH,
            {
                "strategy": strategy_name,
                "task_id": str(task.id),
                "seed": _json_int(seed),
                "round": _json_int(round_idx),
                "row_idx": int(idx),
                "labels": {col: labels[col] for col in cols},
                "reason": reason,
            },
        )


def _json_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _unique_ints(values) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for raw in values:
        try:
            idx = int(raw)
        except (TypeError, ValueError):
            continue
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out
