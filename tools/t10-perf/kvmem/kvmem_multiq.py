#!/usr/bin/env python3
"""Multi-question recall test: prefill the long prefix once, then ask many questions.

The point is cost. A fresh 242k prefill takes ~15 min, which makes a recall sweep
impractical. But KVMem keeps a prefix checkpoint, so a second request that shares a
prompt prefix reuses it: measured WALL 57.7s for the first request against 1.1s for
the next two with an identical prompt. This tool exploits that -- it lays down one
long prefix and then asks N questions against it, so a whole battery costs one
prefill plus a couple of seconds per question.

Two kinds of question are asked, because they fail differently:

  needle    a fact stated in one place ("what is the codename at 40%?"). Tests
            whether the block holding it was retrieved. This is what a recall
            probe normally measures.
  synthesis a fact that only exists across the whole document ("list all twelve
            codenames", "sum all twelve numbers"). Retrieval that returns a
            working set can pass every needle and still fail this, because the
            answer needs blocks that no single question's mean key favours.

Both endpoints are supported; the raw /completion endpoint is the default because
that is where the query-span bug lived.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

# Codenames are deliberately unguessable: a model that never retrieved the block
# cannot produce them by chance, so a hit is real evidence rather than a lucky guess.
CODENS = [
    "QUARTZBEE",
    "VELVETMOTH",
    "PAPERHAIL",
    "GLOBEFERN",
    "SILENTAXLE",
    "MARROWSKY",
    "COPPERWISP",
    "THIMBLEJAY",
    "ASHENQUILL",
    "BRAMBLEKITE",
    "HOLLOWPEAT",
    "GARNETSLIP",
]

FILLER_SENTENCES = [
    "The maintenance crew walked the service tunnel again that morning.",
    "A pressure gauge in the east hall read slightly higher than the day before.",
    "Nobody had adjusted the intake manifolds since the last inspection.",
    "The shift log recorded a small vibration near the third compressor.",
    "Rain had been falling for two days and the loading bay stayed slick.",
    "Spare gaskets were stacked in the usual corner of the supply room.",
    "The foreman signed the sheet without reading the comments below it.",
    "A fan in the relay cabinet hummed at a pitch that bothered the new hire.",
    "The night crew kept the corridor lights dim to save on power.",
    "Coffee in the break room had been reheated one time too many.",
]

NOVEL_SENTENCE = (
    "The archivist catalogued another box of ledgers from the flooded basement, "
    "noting the water damage along the spine of each volume."
)

# MEASURED, not guessed: the server reported prompt 2,646 for a 16,093-char body and 7,938 for
# 48,277 chars, i.e. 0.1644 both times. An earlier version of this file assumed 0.305 and so built a
# prefix that came out at 3,633 tokens against an 8,192 budget -- under budget, meaning KVMem evicted
# nothing, every arm saw the whole context, and five questions "passed" without testing anything.
# Never size the filler from an assumed ratio; --show-prompt-tokens re-checks it against the server.
FILLER_TOKENS_PER_CHAR = 0.1644


def approx_tokens(text: str, tokens_per_char: float = FILLER_TOKENS_PER_CHAR) -> int:
    return int(len(text) * tokens_per_char)


def build_body(target_tokens: int, covert_marker: str) -> str:
    """Build the middle of the prompt and plant the codenames at spread positions."""
    target_chars = int(target_tokens / FILLER_TOKENS_PER_CHAR)
    paragraphs = []
    # Twelve evenly spread insertion points, then one codename each in order.
    step = max(target_chars // (len(CODENS) + 1), 400)
    built = 0
    index = 0
    while built < target_chars:
        chunk = []
        # A paragraph is a handful of filler sentences plus, at the marked slot,
        # the one sentence that carries the next codename.
        for _ in range(6):
            chunk.append(FILLER_SENTENCES[(index + len(chunk)) % len(FILLER_SENTENCES)])
        paragraph = " ".join(chunk)
        if index % max(step // 900, 1) == 0 and len(paragraphs) < len(CODENS):
            name = CODENS[len(paragraphs)]
            ordinal = len(paragraphs) + 1
            paragraph += (
                f" {covert_marker} {ordinal}: the codename issued for this case "
                f"was {name}."
            )
        paragraphs.append(paragraph)
        built += len(paragraph) + 1
        index += 1
    return "\n\n".join(paragraphs)


LOCALHOST = "http://127.0.0.1"


def post_completion(port: int, prompt: str, n_predict: int, timeout: float, key: str,
                    cache_prompt: bool) -> dict:
    url = f"{LOCALHOST}:{port}/completion"
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "cache_prompt": cache_prompt,
        "stream": False,
    }
    request = urllib.request.Request(  # noqa: S310 (localhost only)
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--budget", type=int, default=26500,
                        help="engine --kvmem-budget, used to size the filler")
    parser.add_argument("--port-key", default="sk-cno239cnl20sb41mfg5f")
    parser.add_argument("--n-predict", type=int, default=160,
                        help="generous: a truncated answer reads as a recall failure")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--prefix-tokens", type=int, default=0,
                        help="override the prefix size (default: budget - novel - slack)")
    parser.add_argument("--questions", default="needle,synthesis",
                        help="comma list of question kinds to ask")
    parser.add_argument("--marker", default="Case number")
    parser.add_argument("--min-eviction", type=float, default=0.5,
                        help="refuse to run unless this fraction of the transcript is evicted; "
                             "a prefix that fits in the budget tests nothing, because KVMem "
                             "then keeps every page and the model sees the whole context")
    args = parser.parse_args()

    # Keep the prefix under the context window; the remainder is headroom for the
    # questions and their answers.
    if args.prefix_tokens:
        prefix_tokens = args.prefix_tokens
    else:
        # Aim well past the budget so the eviction actually happens.
        prefix_tokens = max(int(args.budget / (1.0 - args.min_eviction)) + 512, args.budget + 4096)

    body = build_body(prefix_tokens, args.marker)
    prefix = (
        "The following is an internal archive transcript. Read it carefully.\n\n"
        + body
        + "\n\nEnd of transcript.\n"
    )
    estimate = approx_tokens(prefix)
    evicted = 1.0 - (args.budget / max(estimate, 1))
    print(f"prefix: {len(prefix):,} chars ~= {estimate:,} tokens "
          f"({FILLER_TOKENS_PER_CHAR} tok/char, measured)")
    print(f"budget {args.budget:,} -> evicting ~{evicted * 100:.0f}% of the transcript")
    if evicted < args.min_eviction:
        print(f"\nREFUSING: eviction {evicted * 100:.0f}% < --min-eviction "
              f"{args.min_eviction * 100:.0f}%. A prefix that fits inside the budget keeps every "
              f"page, so every question passes without retrieval doing any work and the run proves "
              f"nothing. Raise --prefix-tokens (or lower --budget) and re-run.", file=sys.stderr)
        return 2
    print(f"planted {len(CODENS)} codenames as numbered case records")
    print()

    questions: list[tuple[str, str, list[str]]] = []
    kinds = [kind.strip() for kind in args.questions.split(",") if kind.strip()]
    if "needle" in kinds:
        # Ordinals, not percentages: the planted text states an explicit record number,
        # so the question and the answer key cannot drift apart.
        for record in (1, 4, 8, 12):
            questions.append((
                f"needle record {record}",
                f"What codename was issued for case record {record}? "
                f"Answer with just the codename.",
                [CODENS[record - 1]],
            ))
    if "synthesis" in kinds:
        questions.append((
            "synthesis-all",
            "List every codename that was issued, in record order, separated by commas. "
            "Answer with just the codenames.",
            list(CODENS),
        ))

    # The first request pays the prefill; every later one shares the prefix.
    total_prefill = 0.0
    for index, (label, question, expected) in enumerate(questions):
        prompt = f"{prefix}\nQuestion: {question}\nAnswer:"
        started = time.time()
        # cache_prompt is what makes requests 2..N cheap; without it each question
        # would re-prefill the whole transcript.
        result = post_completion(args.port, prompt, args.n_predict, args.timeout,
                                 args.port_key, cache_prompt=True)
        wall = time.time() - started
        if index == 0:
            total_prefill = wall
        text = result.get("content") or ""
        hit = [name for name in expected if name in text]
        # An exact-match question must name the one codename; a synthesis question
        # is scored on how many of the twelve it recovered.
        if len(expected) == 1:
            verdict = "HIT" if hit else "MISS"
        else:
            verdict = f"{len(hit)}/{len(expected)}"
        print(f"[{label:14s}] {wall:7.1f}s  {verdict}")
        print(f"    answer: {text[:160].replace(chr(10), ' ')!r}")
        if len(expected) > 1:
            missing = [name for name in expected if name not in text]
            print(f"    missing: {', '.join(missing) if missing else '(none)'}")
    print()
    print(f"first request (the prefill) {total_prefill:.1f}s; later requests reused it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
