#!/usr/bin/env python3
"""sample.py <port> <max_tokens> [--temp T] [--top-p P] [--top-k K] [--min-p M]

Two fixed generations, for regression comparison. Speed numbers say nothing about whether a model
still thinks, so this asks a question that needs reasoning and records both the reasoning and the
answer; a later build is compared against the recorded text.

The chat endpoint is used rather than /completion: these are instruct models, and a raw prompt is
continued as a document instead of being answered. The chat endpoint reads `max_tokens`; `n_predict`
is silently ignored there and the generation runs to the context limit.

Thinking stays on, as the operator runs it. A thinking tower emits its reasoning into a separate
field, so the answer is the text after the last `</think>` -- a truncated reasoning block would
otherwise read as a wrong answer. The small towers spend the whole budget inside their reasoning on
the harder question, which is recorded as `THINKING` rather than treated as a failure: the digest of
the reasoning is still a valid fingerprint, and the second, trivial question gives a graded answer
for every tower.
"""
import argparse
import hashlib
import json
import sys
import urllib.request

QUESTIONS = (
    (
        "trivial",
        "What is 2+2? Answer with the number only.",
        "4",
    ),
    (
        "train",
        "A train leaves at 09:15 and travels 168 km at a constant 96 km/h. "
        "It stops for 12 minutes halfway. At what time does it arrive? "
        "Answer with the arrival time only.",
        "11:12",
    ),
)


def ask(port: str, question: str, max_tokens: int, sampling: dict) -> tuple[str, str, int]:
    """Returns (reasoning, answer, generated token count)."""
    body = {
        "messages": [{"role": "user", "content": question}],
        "max_tokens": max_tokens,
        "cache_prompt": False,
        **sampling,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-cno239cnl20sb41mfg5f",
        },
    )
    assert request.full_url.startswith("http://127.0.0.1:"), request.full_url
    with urllib.request.urlopen(request, timeout=3600) as response:  # noqa: S310 (localhost)
        result = json.loads(response.read())
    assert result is not None
    choice = (result.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    timings = result.get("timings") or {}
    return (
        message.get("reasoning_content") or "",
        message.get("content") or "",
        int(timings.get("predicted_n") or 0),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("port")
    parser.add_argument("max_tokens", type=int)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    args = parser.parse_args()

    sampling: dict = {"temperature": args.temp}
    if args.top_k is not None:
        sampling["top_k"] = args.top_k
    if args.top_p is not None:
        sampling["top_p"] = args.top_p
    if args.min_p is not None:
        sampling["min_p"] = args.min_p
    # repeat_penalty is deliberately absent: NInfer refuses any value but its neutral 1.0
    # (src/serve/text_completion.cpp require_neutral_controls), so the preset's 1.1 cannot be sent.

    combined = hashlib.sha256()
    for label, question, expected in QUESTIONS:
        reasoning, answer, generated = ask(args.port, question, args.max_tokens, sampling)
        tail = answer.rsplit("</think>", 1)[-1].strip() if "</think>" in answer else answer.strip()
        if expected in tail:
            verdict = "HIT"
        elif not tail:
            verdict = "THINKING"
        else:
            verdict = "MISS"
        combined.update((reasoning + "\x00" + answer).encode())
        last = tail.splitlines()[-1][:64] if tail else "(no answer, reasoning only)"
        print(
            f"  SAMPLE {label} {verdict} gen {generated} tok "
            f"reasoning {len(reasoning)} answer {len(answer)} | {last}"
        )
    print(f"  SAMPLE digest sha256:{combined.hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
