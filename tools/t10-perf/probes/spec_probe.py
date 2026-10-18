#!/usr/bin/env python3
"""spec_probe.py <port> <n_predict> [--spec NAME] [--drafts N]
Realistic multi-language code prompt, greedy, no repetition -> drafting must actually predict."""
import json, sys, time, urllib.error, urllib.request

port = sys.argv[1]; n_predict = int(sys.argv[2])
PROMPT = '''Write a Python function that parses an ISO-8601 duration string like "P1DT2H30M" into
total seconds. Handle years of 365 days and months of 30 days. Raise ValueError on malformed
input. Then write the TypeScript equivalent and a Rust one, with the same validation rules.

```python
'''
body = json.dumps({"prompt": PROMPT, "n_predict": n_predict, "temperature": 0.0,
                   "cache_prompt": False}).encode()
url = f"http://127.0.0.1:{port}/completion"
assert url.startswith("http://127.0.0.1:")
deadline = time.monotonic() + 300
while True:
    req = urllib.request.Request(url, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer sk-cno239cnl20sb41mfg5f"})
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:  # noqa: S310
            res = json.loads(r.read())
        break
    except urllib.error.HTTPError as e:
        if e.code != 503 or time.monotonic() > deadline: raise
        time.sleep(2)
t = res.get("timings") or {}
print(f"  prompt {t.get('prompt_n')} | gen {t.get('predicted_n')} | "
      f"DECODE {t.get('predicted_per_second',0):.2f} tok/s", end="")
if "draft_n" in t:
    print(f" | draft accepted {t['draft_n_accepted']}/{t['draft_n']}"
          f" = {100.0*t['draft_n_accepted']/max(1,t['draft_n']):.1f}%")
else:
    print()
