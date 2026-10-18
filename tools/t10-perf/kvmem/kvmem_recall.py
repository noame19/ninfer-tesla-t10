#!/usr/bin/env python3
"""kvmem_recall.py -- does a KVMem window still retrieve a probe it evicted?

Puts three unique probe strings at known depths of the context, asks for them back in order,
and reports HIT/MISS per probe. A HIT is possible only if KVMem's retrieval brought the block
back, because the window (--kvmem-budget) is far smaller than the context.

  kvmem_recall.py --endpoint completion --budget 242000        # raw /completion
  kvmem_recall.py --endpoint chat --budget 242000              # /v1/chat/completions

With --endpoint completion the whole prompt is the query (frontend.cpp:1347 sets
memory_query = {0, n}); with chat, the last user turn is the query (chat_template.cpp:438).
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

FILLER = os.environ.get("NINFER_T10_RECALL_FILLER", "")
NOVEL  = os.environ.get("NINFER_T10_RECALL_NOVEL", "")
NOVEL_TOKENS = 82767  # the novel's own length, in tokens; the 242k probe uses all of it
PROBES = [("BEGIN", "ZXQ-7741", 0.01), ("MID", "ZXQ-2288", 0.50), ("END", "ZXQ-9034", 0.98)]
QUESTION = ("上文是一部小说，其中一共出现了三处《内部记录》，每处都写了一个探针编号。"
            "请按它们在小说中出现的先后顺序，逐行回答这三个编号，格式：BEGIN=编号，MID=编号，END=编号。")


def _corpus(path: str, variable: str) -> str:
    """Read a probe corpus, naming the environment variable when it is not configured.

    The paths live in env.sh, which every rig script sources. Reading them here instead of
    hard-coding them keeps the corpora out of the repository: the filler is generated, and the
    narrative half is third-party text.
    """
    if not path:
        raise SystemExit(f"{variable} is not set; source tools/t10-perf/env.sh first")
    return Path(path).read_text(encoding="utf-8")


def build_context(budget, novel_first=False):
    """Filler first, then the novel carrying the probes, so 'BEGIN' is the deepest probe.

    The split is fixed in tokens rather than characters: the novel takes a fifth of the budget
    (capped at its own length) and the filler takes the rest, so --budget scales the whole thing.
    """
    filler = _corpus(FILLER, "NINFER_T10_RECALL_FILLER")
    novel  = _corpus(NOVEL, "NINFER_T10_RECALL_NOVEL")
    # 0.602 tok/char for the novel, 0.305 for the filler (measured on this tokenizer).
    novel_tokens = min(NOVEL_TOKENS, max(int(budget * 0.2), 2000))
    novel = (novel * 4)[:int(novel_tokens / 0.602)]
    parts, last = [], 0
    for _, pid, frac in PROBES:
        at = int(len(novel) * frac)
        parts.append(novel[last:at])
        parts.append(f"\n\n【内部记录】此处探针编号是 {pid}。\n\n")
        last = at
    parts.append(novel[last:])
    novel = "".join(parts)
    remaining = max(budget - novel_tokens - 600, 0)
    filler_chars = int(remaining / 0.305)
    filler = (filler * (filler_chars // len(filler) + 2))[:filler_chars]
    if novel_first:
        # Probes land in the evicted prefix: with the novel in front, BEGIN sits at ~0.2% of the
        # prompt and the retained tail ends far above it, so a HIT requires real retrieval.
        return novel + "\n\n" + filler, len(filler), len(novel)
    return filler + "\n\n" + novel, len(filler), len(novel)


def post(port, key, path, payload, timeout):
    body = json.dumps(payload).encode()
    url = f"http://127.0.0.1:{port}{path}"
    if not url.startswith("http://127.0.0.1:"):  # the scheme and host are ours, never the caller's
        raise ValueError(url)
    req = urllib.request.Request(url, data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:  # noqa: S310 (localhost only)
            return json.load(response), time.time() - started, None
    except urllib.error.HTTPError as error:
        return None, time.time() - started, error.read()[:300].decode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--key", default="sk-cno239cnl20sb41mfg5f")
    parser.add_argument("--budget", type=int, default=242000, help="context tokens to build")
    parser.add_argument("--endpoint", choices=["completion", "chat"], default="completion")
    parser.add_argument("--n-predict", type=int, default=96)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--novel-first", action="store_true",
                        help="put the probe-carrying novel at the START, so the probes are evicted")
    args = parser.parse_args()

    context, filler_chars, novel_chars = build_context(args.budget, args.novel_first)
    if args.endpoint == "completion":
        payload = {"prompt": context + "\n\n" + QUESTION, "n_predict": args.n_predict,
                   "temperature": 0.0, "top_k": 1}
        path = "/completion"
    else:
        payload = {"messages": [{"role": "system", "content": context},
                                {"role": "user", "content": QUESTION}],
                   "n_predict": args.n_predict, "temperature": 0.0}
        path = "/v1/chat/completions"

    result, wall, error = post(args.port, args.key, path, payload, args.timeout)
    if error is not None:
        print(f"HTTP {error}")
        return 1
    assert result is not None
    answer = result.get("content") or result["choices"][0]["message"].get("content") or ""
    choice = (result.get("choices") or [{}])[0]
    reasoning = (result.get("reasoning_content")
                 or choice.get("message", {}).get("reasoning_content") or "")
    haystack = (answer + reasoning).lower()
    print(f"WALL {wall:.1f}s  (answer {len(answer)} chars, reasoning {len(reasoning)} chars)")
    print("ANSWER:", repr(answer[:300]))
    hits = 0
    for name, probe, _ in PROBES:
        hit = probe.lower() in haystack
        hits += hit
        print(f"  {name} {probe}: {'HIT' if hit else 'MISS'}")
    print(f"SCORE {hits}/3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
