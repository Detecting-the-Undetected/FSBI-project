#!/bin/bash
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
# Time, job-name and log paths are set by launch_pipeline.sh at submit time.
#
# One script, four stages, selected by the STAGE env var:
#   train_sbi : Stage 1, FSBI backbone training (time-boxed via --max-hours)
#   extract   : run the best Stage 1 checkpoint over every video -> embeddings
#   bilstm    : Stage 2, Bi-LSTM training on those embeddings
#   evaluate  : (Celeb-DF only) video-level AUC on the official test list -> eval_results.json
#
# Required env (set by launch_pipeline.sh):
#   STAGE TAG CROPPED_DIR LANDMARK_DIR PROJECT_DIR VENV
# Optional env:
#   EPOCHS BATCH_SIZE MAX_HOURS BILSTM_EPOCHS DEBUG MAX_VIDEOS TEST_LIST FAKE_CROPPED_DIR

set -euo pipefail

module load python/3.11.14
source "$VENV/bin/activate"
cd "$PROJECT_DIR/src"

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BILSTM_EPOCHS="${BILSTM_EPOCHS:-50}"
DEBUG="${DEBUG:-false}"
TEST_LIST="${TEST_LIST:-}"
TL_JSON=""; TL_ARG=""
if [ -n "$TEST_LIST" ]; then TL_JSON=",\n    \"test_list\": \"$TEST_LIST\""; TL_ARG="--test-list $TEST_LIST"; fi
EMB_DIR="$PROJECT_DIR/embeddings/$TAG"
CFG="$PROJECT_DIR/src/configs/sbi/${TAG}_cluster.json"

echo "[PIPELINE] stage=$STAGE tag=$TAG host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

# Most recent Stage 1 run directory for this TAG.
# train_sbi.py names it output/<session>_<config-basename>_<timestamp>/
latest_run_dir() {
    ls -dt output/${TAG}_${TAG}_cluster_*/ 2>/dev/null | head -1
}

# Highest-AUC checkpoint (files are named <epoch>_<auc>_val.tar), else latest.tar
best_checkpoint() {
    local run_dir="$1"
    local best
    best=$(ls "$run_dir"weights/*_val.tar 2>/dev/null \
        | awk -F/ '{n=split($NF,a,"_"); print a[2], $0}' \
        | sort -rn | head -1 | cut -d' ' -f2-)
    if [ -z "$best" ]; then best="${run_dir}weights/latest.tar"; fi
    echo "$best"
}

case "$STAGE" in
  train_sbi)
    # Generate the config here so cluster paths and epoch count are never hand-edited.
    cat > "$CFG" << CFGEOF
{
    "epoch": $EPOCHS,
    "batch_size": $BATCH_SIZE,
    "image_size": 380,
    "cropped_dir": "$CROPPED_DIR",
    "landmark_dir": "$LANDMARK_DIR",
    "debug": $DEBUG$(printf "$TL_JSON")
}
CFGEOF
    cat "$CFG"
    python train_sbi.py "$CFG" -n "$TAG" --max-hours "$MAX_HOURS"
    ;;

  extract)
    RUN_DIR=$(latest_run_dir)
    [ -n "$RUN_DIR" ] || { echo "[PIPELINE][ERROR] no Stage 1 run dir found for $TAG"; exit 1; }
    CKPT=$(best_checkpoint "$RUN_DIR")
    [ -f "$CKPT" ] || { echo "[PIPELINE][ERROR] checkpoint not found: $CKPT"; exit 1; }
    echo "[PIPELINE] using checkpoint: $CKPT"
    EXTRA=""
    if [ -n "${MAX_VIDEOS:-}" ]; then EXTRA="--max-videos $MAX_VIDEOS"; fi
    python extract_embeddings.py --checkpoint "$CKPT" \
        --cropped-dir "$CROPPED_DIR" --landmark-dir "$LANDMARK_DIR" \
        --output-dir "$EMB_DIR" --n-frames 32 $EXTRA
    ;;

  bilstm)
    python train_bilstm.py --embedding-dir "$EMB_DIR" -n "$TAG" \
        --epochs "$BILSTM_EPOCHS" --batch-size 32 --max-hours "${BILSTM_MAX_HOURS:-4}" $TL_ARG
    ;;

  evaluate)
    [ -n "$TEST_LIST" ] && [ -n "${FAKE_CROPPED_DIR:-}" ] || { echo "[PIPELINE][ERROR] evaluate needs TEST_LIST and FAKE_CROPPED_DIR"; exit 1; }
    S1_RUN=$(latest_run_dir)
    B_RUN=$(ls -dt output/bilstm_${TAG}_*/ 2>/dev/null | head -1)
    [ -n "$S1_RUN" ] && [ -n "$B_RUN" ] || { echo "[PIPELINE][ERROR] missing Stage 1 or Bi-LSTM run dir for $TAG"; exit 1; }
    S1_CKPT=$(best_checkpoint "$S1_RUN"); B_CKPT=$(best_checkpoint "$B_RUN")
    echo "[PIPELINE] stage1=$S1_CKPT bilstm=$B_CKPT"
    python evaluate_video_auc.py --stage1-ckpt "$S1_CKPT" --bilstm-ckpt "$B_CKPT" \
        --test-list "$TEST_LIST" --real-cropped-dir "$CROPPED_DIR" \
        --fake-cropped-dir "$FAKE_CROPPED_DIR" --hidden-dim 256 \
        --output-json "$PROJECT_DIR/logs/${TAG}_eval_results.json"
    ;;

  *)
    echo "[PIPELINE][ERROR] unknown STAGE='$STAGE'"; exit 1 ;;
esac

echo "[PIPELINE] stage $STAGE finished OK"
