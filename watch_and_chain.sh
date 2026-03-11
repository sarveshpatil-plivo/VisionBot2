#!/usr/bin/env bash
# Monitors 2025 pipeline. When step 1 (fetch) is done, kills it and restarts
# with --stop-after-summarize. Halts after summarization — waits for manual approval.

WORKDIR="/Users/sarvesh.patil/Desktop/vision-resolve"
LOG="$WORKDIR/pipeline_chain.log"
CHECKPOINT="$WORKDIR/ingestion/raw_tickets_2025-01-01.jsonl"
PID_2025=81367
CUTOFF_DATE="2026-02-01"  # last ticket date >= this means step 1 is nearly done

echo "[$(date)] Watcher started. Monitoring PID $PID_2025 (2025 fetch)..." | tee -a "$LOG"

step1_done=false

while kill -0 $PID_2025 2>/dev/null; do
    COUNT=$(wc -l < "$CHECKPOINT" 2>/dev/null | tr -d ' ' || echo 0)
    LAST_DATE=$(tail -1 "$CHECKPOINT" 2>/dev/null | python3 -c "import json,sys; t=json.loads(sys.stdin.read().strip()); print(t.get('updated_at','?')[:10])" 2>/dev/null || echo "unknown")
    echo "[$(date)] Fetching... $COUNT tickets, last: $LAST_DATE" | tee -a "$LOG"

    # Step 1 done when last ticket date >= cutoff
    if [[ "$LAST_DATE" > "$CUTOFF_DATE" || "$LAST_DATE" == "$CUTOFF_DATE" ]]; then
        echo "[$(date)] Step 1 complete — last ticket $LAST_DATE >= $CUTOFF_DATE. Killing PID $PID_2025..." | tee -a "$LOG"
        kill $PID_2025 2>/dev/null
        sleep 5
        step1_done=true
        break
    fi

    sleep 120
done

if ! $step1_done; then
    echo "[$(date)] PID $PID_2025 exited on its own (may have auto-proceeded past step 1)." | tee -a "$LOG"
fi

COUNT=$(wc -l < "$CHECKPOINT" 2>/dev/null | tr -d ' ' || echo 0)
echo "[$(date)] Checkpoint has $COUNT tickets. Starting summarization (--stop-after-summarize)..." | tee -a "$LOG"

cd "$WORKDIR"
export $(grep -v '^#' .env | xargs) 2>/dev/null || true

"$WORKDIR/.venv/bin/python3" -m ingestion.run_pipeline --since 2025-01-01 --stop-after-summarize >> "$LOG" 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "" | tee -a "$LOG"
    echo "========================================" | tee -a "$LOG"
    echo "[$(date)] DONE — 2025 summarization complete." | tee -a "$LOG"
    echo "Waiting for your approval to continue." | tee -a "$LOG"
    echo "Next step: run the following when you're back with internet:" | tee -a "$LOG"
    echo "  cd $WORKDIR && python3 -m ingestion.run_pipeline --since 2021-01-01" | tee -a "$LOG"
    echo "========================================" | tee -a "$LOG"
else
    echo "[$(date)] ERROR — pipeline exited with code $EXIT_CODE. Check log above." | tee -a "$LOG"
fi
