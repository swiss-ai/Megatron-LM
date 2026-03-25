#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --job-name=validate-ckpts
#SBATCH --environment=/iopsstor/scratch/cscs/ahernnde/ncg_new_v2.toml
#SBATCH --time=02:00:00
#SBATCH --output=/users/%u/megatron-repo/logs/slurm/validate_ckpts_%j.out
#SBATCH --error=/users/%u/megatron-repo/logs/slurm/validate_ckpts_%j.err
#SBATCH --account=infra01

set -euo pipefail

CKPT_BASE="/users/rkreft/megatron-repo/logs/Meg-Runs/apertus_image_extension"
PASSED=0
FAILED=0
TOTAL=0

echo "============================================"
echo "SFT Checkpoint Validation"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "============================================"

for run_dir in "${CKPT_BASE}"/*SFT*/checkpoints; do
    [ -d "$run_dir" ] || continue
    run_name=$(basename "$(dirname "$run_dir")")

    for iter_dir in "${run_dir}"/iter_*; do
        [ -d "$iter_dir" ] || continue
        iter_name=$(basename "$iter_dir")
        TOTAL=$((TOTAL + 1))

        echo ""
        echo "--- [$TOTAL] ${run_name} / ${iter_name} ---"

        # 1. Check metadata.json exists and is valid JSON
        if [ ! -f "${iter_dir}/metadata.json" ]; then
            echo "  FAIL: metadata.json missing"
            FAILED=$((FAILED + 1))
            continue
        fi
        if ! python3 -c "import json; json.load(open('${iter_dir}/metadata.json'))" 2>/dev/null; then
            echo "  FAIL: metadata.json is not valid JSON"
            FAILED=$((FAILED + 1))
            continue
        fi
        echo "  OK: metadata.json valid"

        # 2. Check common.pt is loadable
        if [ ! -f "${iter_dir}/common.pt" ]; then
            echo "  FAIL: common.pt missing"
            FAILED=$((FAILED + 1))
            continue
        fi
        if ! python3 -c "import torch; torch.load('${iter_dir}/common.pt', map_location='cpu', weights_only=False)" 2>/dev/null; then
            echo "  FAIL: common.pt cannot be loaded"
            FAILED=$((FAILED + 1))
            continue
        fi
        echo "  OK: common.pt loadable"

        # 3. Check all .distcp shards are non-empty and readable
        shard_count=$(ls "${iter_dir}"/*.distcp 2>/dev/null | wc -l)
        empty_shards=$(find "${iter_dir}" -name "*.distcp" -empty 2>/dev/null | wc -l)
        if [ "$shard_count" -eq 0 ]; then
            echo "  FAIL: no .distcp shard files found"
            FAILED=$((FAILED + 1))
            continue
        fi
        if [ "$empty_shards" -gt 0 ]; then
            echo "  FAIL: ${empty_shards} empty .distcp shards"
            FAILED=$((FAILED + 1))
            continue
        fi
        echo "  OK: ${shard_count} shards, none empty"

        # 4. Use torch.distributed.checkpoint to validate metadata + shard consistency
        result=$(python3 -c "
import sys
from torch.distributed.checkpoint import FileSystemReader

reader = FileSystemReader('${iter_dir}')
metadata = reader.read_metadata()
n_entries = len(metadata.state_dict_metadata)

# Verify planner_data (shard mapping) references existing files
import os
files_on_disk = set(os.listdir('${iter_dir}'))
files_in_metadata = set()
for sd in metadata.planner_data.values():
    for wi in sd:
        files_in_metadata.add(wi.storage_data.storage.fqn)
missing = files_in_metadata - files_on_disk
if missing:
    print(f'FAIL: {len(missing)} shard files referenced in metadata but missing on disk: {list(missing)[:5]}')
    sys.exit(1)
print(f'OK: metadata consistent, {n_entries} state_dict entries, {len(files_in_metadata)} referenced shard files all present')
" 2>&1) || true

        if echo "$result" | grep -q "^FAIL"; then
            echo "  $result"
            FAILED=$((FAILED + 1))
            continue
        elif echo "$result" | grep -q "^OK"; then
            echo "  $result"
        else
            echo "  FAIL: unexpected error during metadata validation:"
            echo "  $result" | head -5
            FAILED=$((FAILED + 1))
            continue
        fi

        PASSED=$((PASSED + 1))
    done
done

echo ""
echo "============================================"
echo "Validation complete: ${TOTAL} checkpoints checked"
echo "  PASSED: ${PASSED}"
echo "  FAILED: ${FAILED}"
echo "End time: $(date)"
echo "============================================"

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi