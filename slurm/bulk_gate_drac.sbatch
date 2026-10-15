#!/bin/bash
#SBATCH --job-name=what-to-id-bulk-gate
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

# Submit with --dependency=afterok:PILOT_JOB. Only a passing gate releases bulk work.
set -euo pipefail
RUN="${1:?pilot run directory required}"
module load StdEnv/2023 python/3.11
virtualenv --no-download "$SLURM_TMPDIR/env"
# shellcheck source=/dev/null
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index numpy pandas requests scipy
export PYTHONPATH="$RUN/src"
python "$RUN/benchmarks/verify_bulk_pilot.py" \
    --manifest "$RUN/pilot/manifest.json" --results "$RUN/results" \
    --reference "$RUN/inputs/cpu-reference-512.npz" \
    --photo-hashes "$RUN/inputs/cpu-photo-hashes.json" --out "$RUN/pilot-gate.json"
