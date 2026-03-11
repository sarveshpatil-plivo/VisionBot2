#!/usr/bin/env bash
# Watches 2022 fetch (PID 6509) and notifies when done. Does nothing else.
LOG="/Users/sarvesh.patil/Desktop/vision-resolve/watch_fetch.log"
PID=6509

echo "[$(date)] Watching fetch PID $PID..." | tee -a "$LOG"

while kill -0 $PID 2>/dev/null; do
    COUNT=$(wc -l < /Users/sarvesh.patil/Desktop/vision-resolve/ingestion/raw_tickets_2022-01-01.jsonl | tr -d ' ')
    LAST=$(tail -1 /Users/sarvesh.patil/Desktop/vision-resolve/ingestion/raw_tickets_2022-01-01.jsonl | python3 -c "import json,sys; t=json.loads(sys.stdin.read().strip()); print(t.get('updated_at','?')[:10])" 2>/dev/null)
    echo "[$(date)] $COUNT tickets — last: $LAST" | tee -a "$LOG"
    sleep 300
done

COUNT=$(wc -l < /Users/sarvesh.patil/Desktop/vision-resolve/ingestion/raw_tickets_2022-01-01.jsonl | tr -d ' ')
echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "[$(date)] FETCH COMPLETE — $COUNT tickets total." | tee -a "$LOG"
echo "Waiting for your approval. Nothing else running." | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
