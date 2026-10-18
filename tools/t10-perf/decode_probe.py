#!/usr/bin/env python3
"""decode_probe.py <port> <n_predict> [prompt_tokens] [--ready-timeout S]

Like ab3.py, but it waits for the server to actually accept work (HTTP 200) instead of assuming
"engine ready" in the log means the frontend is serving. T10 runs leave a few seconds of 503
between the log line and the first accepted request.
"""
import json
import sys
import time
import urllib.error
import urllib.request

port = sys.argv[1]
n_predict = int(sys.argv[2])
prompt_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 64
ready_timeout = 300.0

filler = "The archive holds the field notes from the survey. "
prompt = filler * max(1, prompt_tokens // 8)
body = json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0.0,
                   "cache_prompt": False}).encode()
url = f"http://127.0.0.1:{port}/completion"
assert url.startswith("http://127.0.0.1:"), url

deadline = time.monotonic() + ready_timeout
while True:
    request = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer sk-cno239cnl20sb41mfg5f"})
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:  # noqa: S310 localhost
            result = json.loads(response.read())
        break
    except urllib.error.HTTPError as error:
        if error.code != 503 or time.monotonic() > deadline:
            raise
        time.sleep(2)

assert result is not None
timings = result.get("timings") or {}
print(f"  prompt {timings.get('prompt_n')} tok / {timings.get('prompt_ms', 0) / 1000:.1f}s"
      f" | generated {timings.get('predicted_n')} tok / {timings.get('predicted_ms', 0) / 1000:.1f}s"
      f" | DECODE {timings.get('predicted_per_second', 0):.2f} tok/s")
extra = {k: v for k, v in timings.items() if "draft" in k or "accept" in k}
if extra:
    print("  draft:", extra)
