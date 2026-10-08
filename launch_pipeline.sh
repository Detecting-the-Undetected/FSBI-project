#!/bin/bash
# Submits the full chain for ONE dataset: Stage 1 -> embedding extraction -> Bi-LSTM.
# Each job starts only when the previous one finished successfully.
#
# Usage:
#   ./launch_pipeline.sh TAG CROPPED_DIR LANDMARK_DIR EPOCHS STAGE1_HOURS
#
# Example (FF++ and Celeb-DF in parallel, one GPU each):
#   ./launch_pipeline.sh ffpp   ~/data/ffpp/cropped_faces   ~/data/ffpp/landmarks   60 40
#   ./launch_pipeline.sh celebdf ~/data/celebdf/cropped_faces ~/data/celebdf/landmarks 60 40
#
# Smoke test (tiny, run this FIRST):
#   DEBUG=true MAX_VIDEOS=10 BILSTM_EPOCHS=2 ./launch_pipeline.sh smoke <crops> <lms> 2 0.15
#
# Env overrides: PROJECT_DIR (default ~/FSBI-project), VENV (default ~/fsbi_env)

set -euo pipefail
if [ "$#" -ne 5 ]; then
    sed -n '2,15p' "$0"; exit 1
fi

TAG="$1"; CROPPED_DIR="$2"; LANDMARK_DIR="$3"; EPOCHS="$4"; H="$5"
PROJECT_DIR="${PROJECT_DIR:-$HOME/FSBI-project}"
VENV="${VENV:-$HOME/fsbi_env}"
JOB="$PROJECT_DIR/pipeline_job.sbatch"
mkdir -p "$PROJECT_DIR/logs"

# train_sbi.py only checks the time budget at the START of each epoch, so the last
# epoch can overrun --max-hours by up to one epoch. Give SLURM 2h of slack on top.
S1_MIN=$(awk "BEGIN{printf \"%d\", ($H + 2) * 60}")

EXPORTS="ALL,TAG=$TAG,CROPPED_DIR=$CROPPED_DIR,LANDMARK_DIR=$LANDMARK_DIR,PROJECT_DIR=$PROJECT_DIR,VENV=$VENV,EPOCHS=$EPOCHS,MAX_HOURS=$H,DEBUG=${DEBUG:-false},BILSTM_EPOCHS=${BILSTM_EPOCHS:-50}"
if [ -n "${MAX_VIDEOS:-}" ]; then EXPORTS="$EXPORTS,MAX_VIDEOS=$MAX_VIDEOS"; fi

J1=$(sbatch --parsable --job-name="${TAG}_s1_train" --time="$S1_MIN" \
     --output="$PROJECT_DIR/logs/${TAG}_1_train_%j.out" \
     --export="$EXPORTS,STAGE=train_sbi" "$JOB")

J2=$(sbatch --parsable --job-name="${TAG}_s2_extract" --time=300 \
     --dependency=afterok:"$J1" \
     --output="$PROJECT_DIR/logs/${TAG}_2_extract_%j.out" \
     --export="$EXPORTS,STAGE=extract" "$JOB")

J3=$(sbatch --parsable --job-name="${TAG}_s3_bilstm" --time=300 \
     --dependency=afterok:"$J2" \
     --output="$PROJECT_DIR/logs/${TAG}_3_bilstm_%j.out" \
     --export="$EXPORTS,STAGE=bilstm" "$JOB")

echo "Submitted chain for '$TAG':  train=$J1  ->  extract=$J2  ->  bilstm=$J3"
echo "Watch:   squeue -u \$USER"
echo "Logs:    $PROJECT_DIR/logs/${TAG}_*"
echo "Cancel:  scancel $J1 $J2 $J3"
