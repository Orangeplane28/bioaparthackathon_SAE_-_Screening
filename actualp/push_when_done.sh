#!/bin/bash
set -e

echo "[BioChain] Waiting for training to complete..."

# Wait until biochain_model_v3.pt exists and stops growing
while true; do
    if [ -f biochain_model_v3.pt ]; then
        SIZE1=$(stat -c%s biochain_model_v3.pt)
        sleep 30
        SIZE2=$(stat -c%s biochain_model_v3.pt)
        if [ "$SIZE1" -eq "$SIZE2" ]; then
            echo "[BioChain] Model file stable. Training done."
            break
        fi
    else
        echo "[BioChain] Waiting... (no model file yet)"
        sleep 60
    fi
done

echo "[BioChain] Starting git commit..."

cd ~/hackathonbio/esm_repo/actualp   # <-- change to your actual repo path

# Stage everything
git add biochain_ml_v3.py
git add uniprot_cache.json   2>/dev/null || true
git add biochain_crypto.py   2>/dev/null || true
git add biochain_dashboard.jsx 2>/dev/null || true

# Commit
git commit -m "feat: ESM-3 training complete + all fixes

- ESM-3 BFloat16/Float32 dtype fix (.float() after load)
- CV precompute cache: 15 encoding passes -> 1
- Calibration eval fixed for CV mode
- biochain_model_v3.pt: ESM-3 1536-dim, 5-fold CV"

# Push
git push origin main

echo "[BioChain] Done. All pushed."
