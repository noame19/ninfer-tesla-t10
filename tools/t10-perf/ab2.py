import json, time, urllib.request, sys
port = sys.argv[1]; n = int(sys.argv[2]); reps = int(sys.argv[3])
prompt = ("The following is a technical discussion of inference engines on Turing hardware.\n" * reps)
body = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0.0, "top_k": 1}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body, headers={"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=1800) as r: out = json.load(r)
print("WALL", round(time.time()-t0,1), "chars", len(out.get("content","")))
