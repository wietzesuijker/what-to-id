#!/bin/bash
#SBATCH --job-name=what-to-id-backfill
#SBATCH --account=rrg-bengioy-ad
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@180

# Usage: sbatch --array=0-LAST%4 --output=RUN/logs/%A_%a.out \
#   slurm/bulk_array_drac.sbatch RUN MANIFEST MODEL SHARDS_PER_TASK
# LAST is ceil(manifest shard count / SHARDS_PER_TASK) - 1.
set -euo pipefail
RUN="${1:?run directory required}"
MANIFEST="${2:?manifest required}"
MODEL="${3:?model snapshot required}"
PER_TASK="${4:?shards per array task required}"
if [[ ! "$PER_TASK" =~ ^[1-9][0-9]*$ || ! "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]+$ ]]; then
    echo 'positive shards per task and a Slurm array task ID are required' >&2
    exit 2
fi
module load StdEnv/2023 python/3.11
count=$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["shards"]))' "$MANIFEST")
start=$((SLURM_ARRAY_TASK_ID * PER_TASK))
if (( start >= count )); then
    echo "array task begins beyond manifest shard count: $start >= $count" >&2
    exit 2
fi
shards=()
for (( index=start; index<start+PER_TASK && index<count; index++ )); do
    shards+=("$index")
done
exec bash "$RUN/slurm/bulk_drac.sbatch" "$RUN" "$MANIFEST" "$MODEL" "${shards[@]}"
