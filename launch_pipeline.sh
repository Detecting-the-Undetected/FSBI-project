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
# Celeb-DF: also set TEST_LIST=/path/to/List_of_testing_videos.txt to hold out the
#   official test videos (omit for FF++).
#
# Celeb-DF final evaluation (4th job): also set FAKE_CROPPED_DIR=<cropped_faces of the 340 fake test videos>
#   (with TEST_LIST). Result: $PROJECT_DIR/logs/<TAG>_eval_results.json
#
# Env overrides: PROJECT_DIR (default ~/FSBI-project), VENV (default ~/fsbi_env)

set -euo pipefail
if [ "$#" -ne 5 ]; then
    sed -n '2,19p' "$0"; exit 1
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
if [ -n "${TEST_LIST:-}" ]; then EXPORTS="$EXPORTS,TEST_LIST=$TEST_LIST"; fi
if [ -n "${FAKE_CROPPED_DIR:-}" ]; then EXPORTS="$EXPORTS,FAKE_CROPPED_DIR=$FAKE_CROPPED_DIR"; fi

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

JALL="$J1 $J2 $J3"; MSG="train=$J1  ->  extract=$J2  ->  bilstm=$J3"
if [ -n "${TEST_LIST:-}" ] && [ -n "${FAKE_CROPPED_DIR:-}" ]; then
    J4=$(sbatch --parsable --job-name="${TAG}_s4_eval" --time=240 \
         --dependency=afterok:"$J3" \
         --output="$PROJECT_DIR/logs/${TAG}_4_eval_%j.out" \
         --export="$EXPORTS,STAGE=evaluate" "$JOB")
    JALL="$JALL $J4"; MSG="$MSG  ->  evaluate=$J4"
fi
echo "Submitted chain for '$TAG':  $MSG"
echo "Watch:   squeue -u \$USER"
echo "Logs:    $PROJECT_DIR/logs/${TAG}_*"
echo "Cancel:  scancel $JALL"
