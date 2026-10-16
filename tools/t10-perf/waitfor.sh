#!/bin/bash
# waitfor.sh <logfile> [<pattern>] [<timeout_sec>] [<pidfile_or_pid>]
#
# Blocks until the log shows the completion pattern, the job dies, or the timeout expires --
# whichever happens first -- then prints the log. Replaces the "sleep N; cat log" pattern, which
# always costs the full sleep even when the job finished early, and silently loses the result when
# the job has already exited.
#
# Exits: 0 = pattern seen, 1 = job died without the pattern, 2 = timeout (job may still run).
set -u
LOG=${1:?logfile}
PAT=${2:-ALLDONE}
TIMEOUT=${3:-1800}
JOB=${4:-}

START=$(date +%s)
while :; do
  if grep -qE "$PAT" "$LOG" 2>/dev/null; then
    echo "=== $PAT after $(( $(date +%s) - START ))s ==="
    cat "$LOG"
    exit 0
  fi
  if [ -n "$JOB" ] && ! kill -0 "$JOB" 2>/dev/null; then
    # One last look, in case it wrote the pattern and exited between the check and here.
    if grep -qE "$PAT" "$LOG" 2>/dev/null; then
      echo "=== $PAT after $(( $(date +%s) - START ))s (job already exited) ==="
    else
      echo "=== job $JOB exited WITHOUT '$PAT' after $(( $(date +%s) - START ))s ==="
    fi
    cat "$LOG"
    exit 1
  fi
  if [ $(( $(date +%s) - START )) -ge "$TIMEOUT" ]; then
    echo "=== TIMEOUT ${TIMEOUT}s, '$PAT' not seen; job $JOB still alive ==="
    cat "$LOG"
    exit 2
  fi
  sleep 5
done
