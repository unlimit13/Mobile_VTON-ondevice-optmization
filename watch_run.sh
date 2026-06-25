#!/usr/bin/env bash
# Detached, session-independent watcher for the running inference.
# Logs progress + final result to a DURABLE file in the project dir, and
# preserves the inference stdout/stderr log. Log-only (no kill): with zram-only
# swap, a memory wall is handled by the kernel OOM-killer, so no freeze risk.
PROJ=/home/willtek/bootcamp/vton/2026_CVPR_Mobile-VTON
SRC_LOG="$1"                       # path to the live inference log (scratchpad)
PROG="$PROJ/run_progress.log"
SAVED_LOG="$PROJ/run_inferz.log"
OUT="$PROJ/output/single"

PID=$(pgrep -f "inference_lowmem.py" | head -1)
echo "=== watch start $(date '+%F %T')  inference PID=$PID ===" >> "$PROG"

start_epoch=$(date +%s)
while [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; do
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  swused=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print int((t-f)/1024)}' /proc/meminfo)
  et=$(ps -o etime= -p "$PID" 2>/dev/null | tr -d ' ')
  dec=$(grep -ac "\[TiledVaeDecoder\]" "$SRC_LOG" 2>/dev/null)
  echo "$(date +%T) etime=$et avail=${avail}MB swap=${swused}MB decode_reached=$dec" >> "$PROG"
  cp "$SRC_LOG" "$SAVED_LOG" 2>/dev/null
  sleep 30
done

cp "$SRC_LOG" "$SAVED_LOG" 2>/dev/null
echo "=== inference process EXITED $(date '+%F %T') ===" >> "$PROG"
echo "output images:" >> "$PROG"
ls -la "$OUT" 2>/dev/null >> "$PROG"
grep -aiE "Done\.|Saved to|RuntimeError|Error" "$SRC_LOG" 2>/dev/null | grep -v "frame #" | tail -5 >> "$PROG"
echo "=== watcher done ===" >> "$PROG"
