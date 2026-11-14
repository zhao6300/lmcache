# SPDX-License-Identifier: Apache-2.0
"""Reproduce high-concurrency output degeneration on a vLLM endpoint.

The script sends known-answer requests at a configurable concurrency ladder and
classifies every response. It was written to reproduce this symptom on
``DeepSeek-V41-Flash`` served by vLLM: at 48-64 concurrent requests whose prompt
asks the model to remember and echo a short session code, many responses come
back with an empty ``content``, ``finish_reason="length"`` after looping inside
``message.reasoning``, or an invented code that resembles the real one.

Why a ladder: the same prompts are answered correctly at concurrency 1 and
degrade as the batch grows, so a single request cannot reproduce the issue.

Usage::

    python tools/repro_concurrency_degeneration.py \
        --base-url http://localhost:8000/v1 \
        --api-key "$API_KEY" \
        --sessions 64 \
        --concurrency 1 8 16 32 48 64

Modes:

* ``nonce`` (default): every session gets a private 6-character code in its
  system prompt and must echo it. A response is correct only if it contains
  its own code. The script also scans both ``content`` and ``reasoning`` for
  the other sessions' codes, so genuine cross-session bleed is reported
  separately from degeneration.
* ``verbatim``: every session repeats one fixed sentence, removing any
  ambiguity about the expected answer.
* ``math``: a trivial ``2+2`` request, included as a sanity control.
* ``varied``: every session has a different system-prompt wording, topic and
  code, so the degeneration is shown to be independent of sharing a prompt.
* ``arith``: every session gets a different addition, answering with a unique
  number.

The exit status is 1 when any response is classified as not correct, unless
``--allow-degeneration`` is passed.
"""

# Future
from __future__ import annotations

# Standard
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import argparse
import json
import os
import random
import string
import sys
import time
import urllib.error
import urllib.request

DEFAULT_CONCURRENCY = (1, 8, 16, 32, 48, 64)
"""Concurrency ladder that exposes the degradation on the affected stack."""

VERBATIM_SENTENCE = (
    "The quick brown fox jumps over the lazy dog near the river bank at noon."
)

VARIED_TEMPLATES = (
    (
        "You are the assistant for project {topic}. The access token is "
        "{code}. When asked, output only the token.",
        "What is the access token?",
    ),
    (
        "For the conversation about {topic}, remember the passphrase {code}. "
        "Reply with only the passphrase if asked.",
        "What is the passphrase?",
    ),
    (
        "Session topic: {topic}. Credential: {code}. Echo only the credential "
        "on request.",
        "Echo the credential.",
    ),
    (
        "You handle the topic {topic}. The secret word is {code}. Answer only "
        "the secret word.",
        "What is the secret word?",
    ),
)


def _make_code(rng: random.Random) -> str:
    """Return a random six-character uppercase alphanumeric code."""
    alphabet = string.ascii_uppercase + string.digits
    return "".join(rng.choice(alphabet) for _ in range(6))


def _normalise(text: str) -> str:
    """Collapse whitespace so verbatim answers can be compared robustly."""
    return " ".join(text.split())


def _build_cases(mode: str, sessions: int, rng: random.Random) -> list["Case"]:
    """Build one known-answer request per session.

    Args:
        mode: One of ``nonce``, ``verbatim`` or ``math``.
        sessions: Number of distinct sessions.
        rng: Random source for session codes; a fixed seed makes the inputs
            reproducible even though the model output is not.

    Returns:
        One :class:`Case` per session.
    """
    cases: list[Case] = []
    for sid in range(sessions):
        secret = _make_code(rng)
        if mode == "nonce":
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"You are session S{sid}. Your private session code is "
                        f"{secret}. When asked for your session code, reply "
                        "with exactly that code and nothing else."
                    ),
                },
                {"role": "user", "content": "What is your session code?"},
            ]
        elif mode == "verbatim":
            secret = VERBATIM_SENTENCE
            messages = [
                {
                    "role": "user",
                    "content": (
                        "Repeat the following sentence verbatim, with no other "
                        f"text: {VERBATIM_SENTENCE}"
                    ),
                }
            ]
        elif mode == "math":
            secret = "4"
            messages = [
                {
                    "role": "user",
                    "content": "What is 2+2? Answer with only the number.",
                }
            ]
        elif mode == "varied":
            topic = "".join(rng.choice(string.ascii_lowercase) for _ in range(7))
            system_template, question = VARIED_TEMPLATES[
                sid % len(VARIED_TEMPLATES)
            ]
            messages = [
                {
                    "role": "system",
                    "content": system_template.format(topic=topic, code=secret),
                },
                {"role": "user", "content": question},
            ]
        elif mode == "arith":
            left = rng.randrange(10, 999)
            right = rng.randrange(10, 999)
            secret = str(left + right)
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"What is {left} + {right}? Answer with only the number."
                    ),
                }
            ]
        else:
            raise ValueError(f"unknown mode: {mode}")
        cases.append(Case(sid=sid, expected=secret, messages=messages))
    return cases


def _post(
    opener: urllib.request.OpenerDirector,
    url: str,
    api_key: str,
    payload: dict,
    timeout: float,
) -> dict:
    """POST one chat completion and return the decoded JSON body.

    Args:
        opener: urllib opener with proxy handling disabled.
        url: Full chat-completions URL.
        api_key: Bearer token, or an empty string when auth is disabled.
        payload: JSON request body.
        timeout: Per-request timeout in seconds.

    Returns:
        The decoded response body, or ``{"_error": ...}`` when the request
        failed.
    """
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST"
    )
    request.add_header("Content-Type", "application/json")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with opener.open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        return {"_error": f"HTTP {exc.code}: {detail}"}
    except OSError as exc:
        return {"_error": f"{type(exc).__name__}: {exc}"}


def _message_fields(body: dict) -> tuple[str, str, str | None, int, int, int]:
    """Extract content, reasoning, finish reason and token counts.

    Both ``reasoning`` (used by this deployment) and ``reasoning_content``
    (the common OpenAI-compatible name) are read.
    """
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    usage = body.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return (
        content,
        reasoning,
        choice.get("finish_reason"),
        int(usage.get("completion_tokens") or 0),
        int(usage.get("prompt_tokens") or 0),
        int(details.get("cached_tokens") or 0),
    )


def _classify(
    case: "Case",
    body: dict,
    elapsed: float,
    others: dict[int, str],
) -> "Result":
    """Classify one response as correct, empty, error or otherwise wrong.

    ``finish_reason`` is stored on the result so truncated/looping responses
    can be counted separately. Cross-session bleed is tracked separately too:
    it is reported when another session's expected code appears in
    ``content`` or ``reasoning``.
    """
    if "_error" in body:
        return Result(
            sid=case.sid,
            expected=case.expected,
            category="error",
            content="",
            reasoning=body["_error"],
            finish_reason=None,
            completion_tokens=0,
            prompt_tokens=0,
            cached_tokens=0,
            elapsed=elapsed,
        )
    content, reasoning, finish, completion, prompt, cached = _message_fields(body)
    text = content.strip()
    ok = _normalise(case.expected) in _normalise(text)
    haystack = _normalise(reasoning + "\n" + text)
    foreign = tuple(
        sid
        for sid, expected in others.items()
        if sid != case.sid and _normalise(expected) in haystack
    )
    if ok:
        category = "ok"
    elif not text:
        category = "empty"
    else:
        category = "other"
    return Result(
        sid=case.sid,
        expected=case.expected,
        category=category,
        content=content,
        reasoning=reasoning,
        finish_reason=finish,
        completion_tokens=completion,
        prompt_tokens=prompt,
        cached_tokens=cached,
        elapsed=elapsed,
        foreign=foreign,
    )


def _run_step(
    opener: urllib.request.OpenerDirector,
    url: str,
    api_key: str,
    model: str,
    mode: str,
    cases: list["Case"],
    concurrency: int,
    max_tokens: int,
    timeout: float,
) -> list["Result"]:
    """Send all cases with at most ``concurrency`` in flight.

    Cross-session detection is enabled only for ``nonce`` mode, where every
    session owns a distinct expected answer.
    """
    others = (
        {case.sid: case.expected for case in cases} if mode == "nonce" else {}
    )
    payload_base = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }

    def run(case: Case) -> Result:
        payload = dict(payload_base, messages=case.messages)
        started = time.perf_counter()
        body = _post(opener, url, api_key, payload, timeout)
        return _classify(case, body, time.perf_counter() - started, others)

    workers = max(1, min(concurrency, len(cases)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, cases))


def _print_report(
    mode: str,
    concurrency: int,
    results: list["Result"],
    max_samples: int = 3,
) -> None:
    """Print one ladder step: category counts and degenerate samples."""
    counts = Counter(result.category for result in results)
    total = len(results)
    foreign = [result for result in results if result.foreign]
    finish_length = sum(1 for r in results if r.finish_reason == "length")
    mean_elapsed = sum(result.elapsed for result in results) / total
    mean_cached = sum(result.cached_tokens for result in results) / total
    mean_completion = (
        sum(result.completion_tokens for result in results) / total
    )
    print(
        f"mode={mode:<8} conc={concurrency:>3}  ok={counts['ok']}/{total}  "
        f"empty={counts['empty']}  other={counts['other']}  "
        f"error={counts['error']}  finish_length={finish_length}  "
        f"foreign={len(foreign)}  cached={mean_cached:.0f}  "
        f"mean_t={mean_elapsed:.2f}s  mean_completion={mean_completion:.0f}"
    )
    for result in results:
        if result.foreign:
            print(
                f"    FOREIGN sid={result.sid} expected={result.expected} "
                f"matches={result.foreign} content={result.content[:60]!r}"
            )
    shown = 0
    for result in results:
        if result.category == "ok" or shown >= max_samples:
            continue
        shown += 1
        print(
            f"    sample[{result.category}] sid={result.sid} "
            f"expected={result.expected} finish={result.finish_reason} "
            f"completion_tokens={result.completion_tokens}"
        )
        print(f"      content  : {result.content[:100]!r}")
        print(f"      reasoning: {result.reasoning[:200]!r}")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce high-concurrency output degeneration on a vLLM "
            "chat-completions endpoint."
        )
    )
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LMCACHE_API_KEY", ""),
        help="Bearer token; defaults to the LMCACHE_API_KEY environment variable.",
    )
    parser.add_argument("--model", default="DeepSeek-V41-Flash")
    parser.add_argument(
        "--mode",
        choices=["nonce", "verbatim", "math", "varied", "arith"],
        default="nonce",
        help="Prompt family used for every session.",
    )
    parser.add_argument("--sessions", type=int, default=64)
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONCURRENCY),
        help="Concurrency ladder; each value is a separate reproduction step.",
    )
    parser.add_argument("--waves", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument(
        "--allow-degeneration",
        action="store_true",
        help="Exit 0 even when degeneration is reproduced.",
    )
    parser.add_argument(
        "--save-json",
        default="",
        help="Write every classified response to this JSON file.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3,
        help="Degenerate samples printed per ladder step.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the concurrency ladder and report degeneration."""
    args = _parse_args()
    api_key = args.api_key or ""
    url = args.base_url.rstrip("/") + "/chat/completions"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    rng = random.Random(args.seed)
    cases = _build_cases(args.mode, args.sessions, rng)

    print(
        f"endpoint={url} model={args.model} mode={args.mode} "
        f"sessions={args.sessions} max_tokens={args.max_tokens} seed={args.seed}"
    )
    any_bad = False
    recorded: list[dict] = []
    for wave in range(args.waves):
        if args.waves > 1:
            print(f"--- wave {wave + 1}/{args.waves} ---")
        for concurrency in args.concurrency:
            results = _run_step(
                opener,
                url,
                api_key,
                args.model,
                args.mode,
                cases,
                concurrency,
                args.max_tokens,
                args.timeout,
            )
            _print_report(args.mode, concurrency, results, args.max_samples)
            any_bad = any_bad or any(
                result.category != "ok" for result in results
            )
            for result in results:
                recorded.append(
                    {
                        "wave": wave,
                        "concurrency": concurrency,
                        "mode": args.mode,
                        **result.__dict__,
                    }
                )

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as handle:
            json.dump(recorded, handle, ensure_ascii=False, indent=2)
        print(f"wrote {len(recorded)} classified responses to {args.save_json}")

    if any_bad:
        print(
            "DEGENERATION REPRODUCED: at least one response was not correct "
            "(empty content, length-terminated loop, or other)."
        )
        return 0 if args.allow_degeneration else 1
    print("no degeneration observed at the tested concurrencies.")
    return 0


@dataclass(frozen=True)
class Case:
    """One request with a known expected answer."""

    sid: int
    expected: str
    messages: list[dict[str, str]]


@dataclass
class Result:
    """Classification of one response."""

    sid: int
    expected: str
    category: str
    content: str
    reasoning: str
    finish_reason: str | None
    completion_tokens: int
    prompt_tokens: int
    cached_tokens: int
    elapsed: float
    foreign: tuple[int, ...] = ()


if __name__ == "__main__":
    sys.exit(main())
