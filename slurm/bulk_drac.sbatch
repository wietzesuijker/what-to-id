#!/bin/bash
#SBATCH --job-name=what-to-id-bulk
#SBATCH --account=rrg-bengioy-ad
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@180

# Usage: sbatch --output=RUN/logs/%A_%a.out slurm/bulk_drac.sbatch \
#   RUN MANIFEST MODEL SHARD [SHARD ...]
# Override --time for the pilot. Stage each assigned shard's photos as
# RUN/photos/SHARD.tar on clusters without compute-node internet.
set -euo pipefail
RUN="${1:?immutable run directory required}"
MANIFEST="${2:?prepared manifest required}"
MODEL="${3:?frozen model snapshot required}"
shift 3
if (( $# == 0 )); then
    echo 'at least one shard index required' >&2
    exit 2
fi

module purge
module load StdEnv/2023 python/3.11 cuda/12.6
virtualenv --no-download "$SLURM_TMPDIR/env"
# shellcheck source=/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --find-links="$RUN/wheels" -r "$RUN/slurm/bulk-drac-requirements.txt"
export PYTHONPATH="$RUN/src"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
mkdir -p "$SLURM_TMPDIR/work" "$SLURM_TMPDIR/model" "$RUN/results"
cp "$MODEL/open_clip_config.json" "$MODEL/open_clip_model.safetensors" "$SLURM_TMPDIR/model/"
pip freeze > "$RUN/logs/environment-${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

args=()
for shard in "$@"; do
    if [[ ! "$shard" =~ ^[0-9]+$ ]]; then
        echo "invalid shard index: $shard" >&2
        exit 2
    fi
    args+=(--shard "$shard")
    if [[ -f "$RUN/photos/$shard.tar" ]]; then
        tar -xf "$RUN/photos/$shard.tar" -C "$SLURM_TMPDIR/work"
    fi
done
# Only explicit online mode may fetch photos. Rorqual uses staged archives.
if [[ "${BULK_ONLINE_PHOTOS:-0}" != 1 ]]; then
    args+=(--offline-photos)
fi
# The worker commits every completed batch to durable scratch. On Slurm's
# warning signal, SIGTERM asks it to stop at the next checkpoint boundary.
child=
trap 'if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; fi' USR1
python -m what_to_id.bulk_embeddings run --manifest "$MANIFEST" \
    --model-snapshot "$SLURM_TMPDIR/model" --work-dir "$SLURM_TMPDIR/work" \
    --out "$RUN/results" --device cuda --batch "${BULK_BATCH_SIZE:-32}" \
    --max-seconds "${BULK_MAX_SECONDS:-9000}" "${args[@]}" &
child=$!
status=0
wait "$child" || status=$?
while kill -0 "$child" 2>/dev/null; do
    status=0
    wait "$child" || status=$?
done
exit "$status"
