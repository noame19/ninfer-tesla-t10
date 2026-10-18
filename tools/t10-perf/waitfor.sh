#!/bin/bash
# waitfor.sh <logfile> [<pattern>] [<timeout_sec>] [<pid>]
#
# One bounded look at a long-running job: it polls until the pattern appears, the job dies, or the
# timeout expires. Use it with a SHORT timeout as a progress check, not as a way to block; the
# calling tool has its own hard limit and kills the call, which looks like a failure even though
# the job is fine.
#
# Exits: 0 = pattern seen, 1 = job gone without the pattern, 2 = still running (not an error).
# Set WAITFOR_TAIL to control how much of the log is printed (default 20 lines).
set -u
LOG=${1:?logfile}
PAT=${2:-ALLDONE}
TIMEOUT=${3:-60}
JOB=${4:-}
TAIL=${WAITFOR_TAIL:-20}

if [ ! -e "$LOG" ]; then
  echo "=== NO LOG: $LOG does not exist (the job may not have started) ==="
  exit 1
fi

show() { tail -n "$TAIL" "$LOG"; }

START=$(date +%s)
while :; do
  if grep -qE "$PAT" "$LOG" 2>/dev/null; then
    echo "=== DONE: '$PAT' after $(( $(date +%s) - START ))s ==="
    show
    exit 0
  fi
  if [ -n "$JOB" ] && ! kill -0 "$JOB" 2>/dev/null; then
    if grep -qE "$PAT" "$LOG" 2>/dev/null; then
      echo "=== DONE: '$PAT' (job $JOB has exited) ==="
      exit 0
    fi
    echo "=== FAILED: job $JOB is gone and '$PAT' never appeared ==="
    show
    exit 1
  fi
  if [ $(( $(date +%s) - START )) -ge "$TIMEOUT" ]; then
    echo "=== RUNNING: '$PAT' not yet; waited ${TIMEOUT}s${JOB:+, job $JOB alive} ==="
    echo "--- progress ($(wc -l < "$LOG") log lines) ---"
    show
    exit 2
  fi
  sleep 3
done
