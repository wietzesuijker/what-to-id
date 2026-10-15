#!/bin/bash
#SBATCH --job-name=what-to-id-bulk-photos
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
# Submit an array with output directed to RUN/logs; compute nodes need photo-host access.
# Usage: sbatch --array=0-N --output=RUN/logs/stage-%A_%a.log \
#   slurm/bulk_stage_drac.sbatch RUN MANIFEST [SHARDS_PER_TASK=16]
set -euo pipefail

RUN="$(realpath "${1:?durable run directory}")"
MANIFEST="$(realpath "${2:?bulk shard manifest}")"
SHARDS_PER_TASK="${3:-16}"
: "${SLURM_TMPDIR:?node-local scratch is required}"
: "${SLURM_ARRAY_TASK_ID:?submit this script as a zero-based array}"
test -d "$RUN/src/what_to_id"
export PYTHONPATH="$RUN/src"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PIP_NO_INDEX=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
cd "$SLURM_TMPDIR"

module load StdEnv/2023 python/3.11
virtualenv --no-download "$SLURM_TMPDIR/bulk-stage-env"
# shellcheck source=/dev/null
source "$SLURM_TMPDIR/bulk-stage-env/bin/activate"
wheel_options=()
if test -d "$RUN/wheels"; then
    wheel_options+=(--find-links "$RUN/wheels")
fi
python -m pip install --no-index "${wheel_options[@]}" \
    numpy pandas requests scipy 'pillow==11.1.0'
python --version
python -m pip list --format=freeze
printf 'job=%s task=%s host=%s shards_per_task=%s\n' \
    "${SLURM_JOB_ID:-unknown}" "$SLURM_ARRAY_TASK_ID" "$(hostname)" "$SHARDS_PER_TASK"

# wait must be interruptible so Slurm's warning reaches the Python staging handler.
child=""
trap 'if test -n "$child"; then kill -USR1 "$child" 2>/dev/null || true; fi' USR1
python - "$RUN" "$MANIFEST" "$SHARDS_PER_TASK" "$SLURM_ARRAY_TASK_ID" <<'PY' &
import hashlib
import json
import os
import signal
import sys
from pathlib import Path

import pandas as pd

from what_to_id.staging import Preempted, _preempt, stage_group


def main():
    run, manifest_path = map(Path, sys.argv[1:3])
    per_task, task = map(int, sys.argv[3:5])
    if per_task <= 0 or task < 0:
        raise ValueError("shards per task must be positive and task ID nonnegative")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    shards = manifest["shards"]
    if not isinstance(shards, list) or not shards:
        raise ValueError("manifest must contain a nonempty shard list")
    if [s["index"] for s in shards] != list(range(len(shards))):
        raise ValueError("manifest shard indices must be contiguous and zero-based")
    start = task * per_task
    assigned = shards[start:start + per_task]
    if not assigned:
        raise ValueError(f"array task {task} is outside the manifest")
    print(json.dumps({
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "task": task, "shards": [s["index"] for s in assigned],
        "model": manifest.get("model"),
    }), flush=True)
    # Validate every assigned CSV before downloading or replacing any archive.
    validated = []
    for shard in assigned:
        source = (manifest_path.parent / shard["file"]).resolve()
        if not source.is_relative_to(manifest_path.parent.resolve()):
            raise ValueError("shard CSV must be inside the manifest directory")
        if hashlib.sha256(source.read_bytes()).hexdigest() != shard["sha256"]:
            raise ValueError(f"shard {shard['index']}: CSV checksum mismatch")
        rows = pd.read_csv(source, dtype={"id": "int64"})
        if not {"id", "photo_url"} <= set(rows):
            raise ValueError("shard CSV requires id and photo_url")
        ids = [int(i) for i in rows["id"]]
        if (
            len(rows) != shard["count"]
            or ids != shard["ids"]
            or len(set(ids)) != len(ids)
            or any(i <= 0 for i in ids)
        ):
            raise ValueError(f"shard {shard['index']}: CSV IDs/count mismatch")
        validated.append((str(shard["index"]), rows))
    out = run / "photos"
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(os.environ["SLURM_TMPDIR"]) / "cache"
    for group, rows in validated:
        print(f"Staging shard {group}: {len(rows)} requested IDs", flush=True)
        # Keep unavailable IDs in the status and GPU failure ledger. One missing
        # photo must not prevent the other photos in an array task being encoded.
        stage_group(out, group, rows, cache, workers=4, fraction=1.0)
        print(f"Finished shard {group}; failures remain in {out / (group + '.json')}", flush=True)
    return 0


signal.signal(signal.SIGUSR1, _preempt)
try:
    raise SystemExit(main())
except Preempted:
    print("Staging incomplete after USR1; archived progress can resume", file=sys.stderr)
    raise SystemExit(75)
PY
child=$!
status=0
wait "$child" || status=$?
while kill -0 "$child" 2>/dev/null; do
    status=0
    wait "$child" || status=$?
done
if (( status == 75 && ${SLURM_RESTART_COUNT:-0} < 3 )); then
    # Resume only the explicit checkpointed time-limit case, never arbitrary errors.
    scontrol requeue "$SLURM_JOB_ID"
fi
exit "$status"
