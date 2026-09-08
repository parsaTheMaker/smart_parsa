#!/usr/bin/env bash
# Copy the completed remote C-core box-mask checkpoints exactly once.
set -euo pipefail

REMOTE="${REMOTE:-parsa@servus06.ge.in.tum.de}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/parsa/smart_parsa}"
LOCAL_ROOT="${LOCAL_ROOT:-/home/parsa/smart_parsa}"
POLL_SECONDS="${POLL_SECONDS:-300}"
STEM="smart-c-core-magnetic-box-masked-box-masked-300ep-ccoremagnetic-s42"
STATUS_PATH="$REMOTE_ROOT/logs/c_core_magnetic_box_masked_servus06/exit_status"

while true; do
  status="$(ssh -o BatchMode=yes "$REMOTE" "cat '$STATUS_PATH' 2>/dev/null" 2>/dev/null || true)"
  if [[ -n "$status" ]]; then
    if [[ "$status" != "0" ]]; then
      printf 'Remote C-core box-mask training exited with status %s; not copying checkpoints.\n' "$status" >&2
      exit 1
    fi

    rsync -a --partial --append-verify \
      "$REMOTE:$REMOTE_ROOT/checkpoints/${STEM}_best.pt" \
      "$LOCAL_ROOT/checkpoints/"
    rsync -a --partial --append-verify \
      "$REMOTE:$REMOTE_ROOT/checkpoints/${STEM}_last.pt" \
      "$LOCAL_ROOT/checkpoints/"
    printf 'Copied completed C-core box-mask checkpoints from %s.\n' "$REMOTE"
    exit 0
  fi
  sleep "$POLL_SECONDS"
done
