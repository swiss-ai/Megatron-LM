#!/bin/bash
# Freeze the two blends of Stage P (data_mix.md): distill/blend.txt for the training cache
# and pruning/blend.txt for the calibration set. A rerun rebuilds them and stops if they
# differ from the frozen files, because the data order of a cache depends on them.
#
#   CACHE_ROOT=/path/to/caches bash examples/apertus_distillation/stage_p_blends.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CACHE_ROOT=${CACHE_ROOT:?set CACHE_ROOT to the directory for both sets}
STORE=/capstor/store/cscs/swissai/infra01
TEXT=$STORE/apertus_1p5/apertus_dataset/prepare-8b/text_stage_3
VISION=$STORE/vision-datasets/Apertus1p5_cooldown_tokenized/8B
REPLAY=$STORE/vision-datasets/Apertus1p5_stage2_tokenized/8B
AUDIO=$STORE/audio-datasets/Apertus1p5_cooldown_tokenized
GROUNDING=$REPLAY/img2txt_blip3_grounding_50m
CAPTIONS="$REPLAY/img2txt_recap_datacomp_1b $REPLAY/img2txt_llava85m_midtrain"

freeze() {
    local output=$1
    shift
    mkdir -p "$(dirname "$output")"
    python3 "$HERE/calibration_blend.py" --output "$output.new" \
        --exclude '*/swiss-caselaw-preprocessed/dump-1/00019_tokens' "$@"
    if [ -f "$output" ]; then
        cmp "$output" "$output.new" || { echo "$output differs from the data on disk" >&2; exit 1; }
        rm -f "$output.new"
    else
        mv "$output.new" "$output"
        chmod a-w "$output"
    fi
}

# Tokens of the plan (stage3_datasets.md): text and audio by size, half of the Stage 2
# replay to grounding.
freeze "$CACHE_ROOT/distill/blend.txt" --group text 112.50 "$TEXT" \
    --group vision_cooldown 34.34 "$VISION" --group grounding 1.91 "$GROUNDING" \
    --group captions 1.91 $CAPTIONS --group audio 46.02 "$AUDIO"

MULTILINGUAL=() OTHER=()
for folder in "$TEXT"/*; do
    case "$(basename "$folder")" in
        *fineweb*|*finetranslations*|*multilingual*|*caselaw*) MULTILINGUAL+=("$folder") ;;
        *) OTHER+=("$folder") ;;
    esac
done
freeze "$CACHE_ROOT/pruning/blend.txt" --group text_multilingual 0.25 "${MULTILINGUAL[@]}" \
    --group text_other 0.25 "${OTHER[@]}" --group vision_cooldown 0.315 "$VISION" \
    --group grounding 0.0175 "$GROUNDING" --group captions 0.0175 $CAPTIONS \
    --group audio 0.15 "$AUDIO"
