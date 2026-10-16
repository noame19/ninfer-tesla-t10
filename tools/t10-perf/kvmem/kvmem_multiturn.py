#!/usr/bin/env python3
"""kvmem_multiturn.py -- does KVMem retrieve a probe that a *previous turn* put deep in history?

The single-request probe (kvmem_recall.py) tests the prefill path. This one tests the path an
agent actually uses: turn 1 fills the context, turn 2 asks a short question about something near
the start of turn 1. Turn 2's query is a few tokens at position ~200k, so the query-probe gate
opens only if cross-request history reuse found the checkpoint -- which is where the ring's
position arithmetic and the restorable-history lookup matter.

  kvmem_multiturn.py --turn1-tokens 200000

Turn 1's body carries one probe per depth, so a partial recovery shows as a gradient:
    DEEP   ~1% into turn 1  (retrieval must bring it back from the Host tier)
    SHAL   ~50%
    TIP    ~99%  (inside the retained tail of turn 1; should always come back)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

FILLER = os.environ.get("NINFER_T10_RECALL_FILLER", "/home/agent/work-rom/ninfer-run/filler.txt")
NOVEL = os.environ.get("NINFER_T10_RECALL_NOVEL", "/home/agent/work-rom/ninfer-run/novel.txt")
NOVEL_TOKENS = 82767
PROBES = [("DEEP", "ZXQ-7741", 0.01), ("SHAL", "ZXQ-2288", 0.50), ("TIP", "ZXQ-9034", 0.98)]
ASK = ("请逐行回答上文三处《内部记录》的探针编号，格式：DEEP=编号，SHAL=编号，TIP=编号。")
# Measured on this tokenizer: novel 0.602 tok/char, filler 0.305 tok/char.
NOVEL_TOKENS_PER_CHAR = 0.602
FILLER_TOKENS_PER_CHAR = 0.305


def build_turn1(tokens: int) -> str:
    """Filler, then the novel carrying the probes -- the novel last, so DEEP is the most distant."""
    filler = Path(FILLER).read_text(encoding="utf-8")
    novel = Path(NOVEL).read_text(encoding="utf-8")
    novel_tokens = min(NOVEL_TOKENS, max(int(tokens * 0.4), 4000))
    novel = (novel * 4)[: int(novel_tokens / NOVEL_TOKENS_PER_CHAR)]
    parts, last = [], 0
    for _, probe, fraction in PROBES:
        at = int(len(novel) * fraction)
        parts.append(novel[last:at])
        parts.append(f"\n\n【内部记录】此处探针编号是 {probe}。\n\n")
        last = at
    parts.append(novel[last:])
    novel = "".join(parts)
    filler_chars = int(max(tokens - novel_tokens - 600, 0) / FILLER_TOKENS_PER_CHAR)
    filler = (filler * (filler_chars // len(filler) + 2))[:filler_chars]
    return (filler + "\n\n" + novel).strip()


def post(port: int, key: str, path: str, payload: dict, timeout: int):
    url = f"http://127.0.0.1:{port}{path}"
    if not url.startswith("http://127.0.0.1:"):  # the scheme and host are ours, never the caller's
        raise ValueError(url)
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (localhost)
            return json.load(response), time.time() - started, None
    except urllib.error.HTTPError as error:
        return None, time.time() - started, error.read()[:300].decode()


def message_of(result: dict | None) -> str:
    if result is None:
        return ""
    if result.get("content"):
        return str(result["content"])
    choices = result.get("choices") or [{}]
    message = choices[0].get("message", {})
    return str(message.get("content", "") + (message.get("reasoning_content") or ""))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--key", default="sk-cno239cnl20sb41mfg5f")
    parser.add_argument("--turn1-tokens", type=int, default=200000)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--drop-history", action="store_true",
                        help="send only the follow-up question (a control: this is NOT a KVMem test, "
                             "because the server then has no history to reuse)")
    args = parser.parse_args()

    turn1 = build_turn1(args.turn1_tokens)
    print(f"turn 1: {len(turn1)} chars (~{args.turn1_tokens} tokens)", flush=True)
    messages = [{"role": "user", "content": turn1}]
    first, wall, error = post(args.port, args.key, "/v1/chat/completions",
                              {"messages": messages, "n_predict": 1, "temperature": 0.0},
                              args.timeout)
    if error is not None:
        print(f"turn 1 HTTP {error}")
        return 1
    print(f"turn 1 accepted in {wall:.0f}s", flush=True)

    if args.drop_history:
        messages = [{"role": "user", "content": ASK}]
    else:
        messages = [*messages,
                    {"role": "assistant", "content": message_of(first)},
                    {"role": "user", "content": ASK}]
    second, wall, error = post(args.port, args.key, "/v1/chat/completions",
                               {"messages": messages, "n_predict": 96, "temperature": 0.0},
                               args.timeout)
    if error is not None:
        print(f"turn 2 HTTP {error}")
        return 1
    text = message_of(second)
    print(f"turn 2 WALL {wall:.1f}s")
    print("ANSWER:", repr(text[:300]))
    hits = 0
    for name, probe, _ in PROBES:
        hit = probe.lower() in text.lower()
        hits += hit
        print(f"  {name} {probe}: {'HIT' if hit else 'MISS'}")
    print(f"SCORE {hits}/3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
