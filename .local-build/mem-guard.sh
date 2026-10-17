#!/bin/sh
# Watch the host's free memory while a build runs and stop the build before the kernel OOM-kills
# something. Run it in the same shell as the build so it can signal the build's process group.
#
#   mem-guard.sh <pid-to-stop> [min-available-mib=1536] [poll-seconds=5]
#
# Exits 0 when the watched pid leaves, 1 when it had to stop the build.
#
# Why this exists: the sm_75 INT8 attention templates make ptxas hold several GiB per translation
# unit. Six of them at once took this 23 GiB host to 0.6 GiB available with 10 GiB of swap in use
# (measured 2026-10-16). Concurrent nvcc jobs are the multiplier, so the guard is a backstop for
# the job count rather than a replacement for it.

set -eu

pid=${1:?usage: mem-guard.sh <pid-to-stop> [min-available-mib] [poll-seconds]}
min_mib=${2:-1536}
poll=${3:-5}

peak_used=0
while kill -0 "$pid" 2>/dev/null; do
    # MemAvailable is the kernel's own estimate of what a new allocation can get without swapping.
    avail_mib=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
    used_mib=$(awk '/^MemTotal:/ {t=$2} /^MemAvailable:/ {a=$2} END {print int((t-a)/1024)}' /proc/meminfo)
    [ "$used_mib" -gt "$peak_used" ] && peak_used=$used_mib
    if [ "$avail_mib" -lt "$min_mib" ]; then
        echo "mem-guard: available ${avail_mib} MiB is below ${min_mib} MiB (peak used ${peak_used} MiB); stopping pid ${pid}" >&2
        kill -TERM "$pid" 2>/dev/null || true
        sleep 5
        kill -KILL "$pid" 2>/dev/null || true
        echo "mem-guard: stopped. peak used ${peak_used} MiB" >&2
        exit 1
    fi
    sleep "$poll"
done
echo "mem-guard: pid ${pid} finished; peak used ${peak_used} MiB" >&2
