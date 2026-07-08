#!/usr/bin/env bash
set -u
RUN_DIR=/root/autodl-tmp/DICE_four_star/runs/vae_full_50
SRC=/root/autodl-tmp/DICE_four_star/src
PY=/root/miniconda3/bin/python
STATUS="$RUN_DIR/status.txt"
cd "$SRC" || exit 1
{
  echo "START $(date -Iseconds)"
  for name in \
    ml10m_longtail_vae \
    ml10m_longtail_vaeips \
    ml10m_longtail_vaedice \
    ml10m_uniform_vae \
    ml10m_uniform_vaeips \
    ml10m_uniform_vaedice \
    ml10m_head_vae \
    ml10m_head_vaeips \
    ml10m_head_vaedice
  do
    cfg="./config/vae/${name}.cfg"
    log="$RUN_DIR/logs/${name}.log"
    echo "RUN $name $(date -Iseconds)"
    "$PY" app.py --flagfile "$cfg" --output="$RUN_DIR/output" --use_visdom=False > "$log" 2>&1
    rc=$?
    echo "DONE $name rc=$rc $(date -Iseconds)"
    if [ "$rc" -ne 0 ]; then
      echo "FAILED $name rc=$rc $(date -Iseconds)"
      exit "$rc"
    fi
  done
  echo "ALL_DONE $(date -Iseconds)"
} >> "$STATUS" 2>&1
