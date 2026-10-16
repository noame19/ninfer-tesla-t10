#!/usr/bin/env python3
"""ab3.py <port> <n_predict> [prompt_tokens] -- one decode measurement.

ab2.py measures prefill by repeating a sentence. This measures decode: a short prompt, then a long
generation. They are different axes of speed and the launch tiers under test are chosen by the
attention width (the number of query rows in flight), which during a plain prompt is the chunk
width and during a decode step is the verify width, so a prefill number cannot see them.
"""
import json
import sys
import urllib.request

port, n_predict = sys.argv[1], int(sys.argv[2])
prompt_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 64

filler = "The archive holds the field notes from the survey. "
prompt = filler * (prompt_tokens // 8)

body = json.dumps({
    "prompt": prompt,
    "n_predict": n_predict,
    "temperature": 0.0,
    "cache_prompt": False,
}).encode()
request = urllib.request.Request(
    f"http://127.0.0.1:{port}/completion", data=body,
    headers={"Content-Type": "application/json", "Authorization": "Bearer sk-cno239cnl20sb41mfg5f"},
)
assert request.full_url.startswith("http://127.0.0.1:"), request.full_url
with urllib.request.urlopen(request, timeout=3600) as response:  # noqa: S310 (localhost only)
    result = json.loads(response.read())
assert result is not None
timings = result.get("timings") or {}
print(f"  prompt {timings.get('prompt_n')} tok / {timings.get('prompt_ms', 0) / 1000:.1f}s"
      f" | generated {timings.get('predicted_n')} tok / {timings.get('predicted_ms', 0) / 1000:.1f}s"
      f" | DECODE {timings.get('predicted_per_second', 0):.2f} tok/s")
