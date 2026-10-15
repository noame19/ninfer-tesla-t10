#!/bin/bash
# measure.sh <tag> <serve-binary> [perplexity-binary]
#
# The two-sided gate every route change in this port uses: prefill speed at 11.2k and 47.6k on the
# T10, plus perplexity on the same build. Pass the perplexity binary to run the quality half; omit
# it to measure speed only (a kernel that changes numerics still needs it).
#
# Recorded baselines (see ../../docs/remaining-sm75-work.md for the current figures):
#   11.2k ~467 tok/s, 47.6k ~386 tok/s, 98k ~299 tok/s, PPL 3.01334 (bit-exact).
set -u
cd "$(dirname "$0")"
TAG=$1; SERVE=$2; PPLBIN=${3:-}
echo "================ $TAG ================"
./ab.sh "$SERVE" 800 16 "$TAG-11k" 2>&1 | grep -E "req#1 done|STARTFAIL"
./ab.sh "$SERVE" 3400 16 "$TAG-47k" 2>&1 | grep -E "req#1 done|STARTFAIL"
if [ -n "$PPLBIN" ]; then
  ./ppl.sh "$PPLBIN" "$TAG" 2>&1 | grep -E "PPL|scored|FATAL"
else
  echo "(no perplexity binary given; quality half skipped)"
fi
echo "================ done ================"
