#!/bin/bash
set -euo pipefail

STAGES=(00_extract 01_split 02_build_features 03_audit_preproc 10_train_gpu 11_train_baselines 12_train_supervised_gpu)

stage_index() {
  local needle="$1"
  local i
  for i in "${!STAGES[@]}"; do
    if [[ "${STAGES[$i]}" == "$needle" ]]; then
      echo "$i"
      return 0
    fi
  done
  return 1
}

read_cfg_value() {
  local key="$1"
  awk -F': *' -v key="$key" '
    $1 ~ "^[[:space:]]*" key "$" {
      val=$2
      sub(/[[:space:]]+#.*$/, "", val)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", val)
      print val
      exit
    }
  ' pipelines/m2/config.yaml
}

START_STAGE="${1:-${START_STAGE:-00_extract}}"
END_STAGE="${2:-${END_STAGE:-12_train_supervised_gpu}}"

if ! START_IDX=$(stage_index "$START_STAGE"); then
  echo "Unknown start stage: $START_STAGE" >&2
  exit 1
fi
if ! END_IDX=$(stage_index "$END_STAGE"); then
  echo "Unknown end stage: $END_STAGE" >&2
  exit 1
fi
if (( START_IDX > END_IDX )); then
  echo "Start stage comes after end stage: $START_STAGE > $END_STAGE" >&2
  exit 1
fi

FEATURE_ID="$(read_cfg_value feature_id)"
RUN_PREFIX="$(read_cfg_value prefix)"

if [[ -z "$FEATURE_ID" ]]; then
  echo "Could not read featurizer.feature_id from pipelines/m2/config.yaml" >&2
  exit 1
fi
if [[ -z "$RUN_PREFIX" ]]; then
  RUN_PREFIX="m2"
fi

STATE_DIR="results/.pipeline_state"
STATE_FILE="${STATE_DIR}/${FEATURE_ID}.latest"
mkdir -p "$STATE_DIR"

if [[ -z "${PIPELINE_RUN_ID:-}" ]]; then
  if [[ "$START_STAGE" == "00_extract" || ! -f "$STATE_FILE" ]]; then
    PIPELINE_RUN_ID="$(date +%Y-%m-%d__%H-%M-%S)__${RUN_PREFIX}__${FEATURE_ID}"
  else
    PREV_RUN_ID="$(cat "$STATE_FILE")"
    PREV_PREFIX="$(printf '%s' "$PREV_RUN_ID" | awk -F'__' '{print $(NF-1)}')"
    PREV_FEATURE_ID="$(printf '%s' "$PREV_RUN_ID" | awk -F'__' '{print $NF}')"
    if [[ "$PREV_PREFIX" != "$RUN_PREFIX" || "$PREV_FEATURE_ID" != "$FEATURE_ID" ]]; then
      PIPELINE_RUN_ID="$(date +%Y-%m-%d__%H-%M-%S)__${RUN_PREFIX}__${FEATURE_ID}"
    else
      PIPELINE_RUN_ID="$PREV_RUN_ID"
    fi
  fi
fi
printf '%s\n' "$PIPELINE_RUN_ID" > "$STATE_FILE"

RUN_DIR="results/runs/${PIPELINE_RUN_ID}"
PBS_LOG_DIR="${RUN_DIR}/pbs_logs"
mkdir -p "$PBS_LOG_DIR"

echo "PIPELINE_RUN_ID=${PIPELINE_RUN_ID}"

prev_job=""
for (( i=START_IDX; i<=END_IDX; i++ )); do
  stage="${STAGES[$i]}"
  script="hpc/m2/${stage}.pbs"
  if [[ ! -f "$script" ]]; then
    echo "Missing stage script: $script" >&2
    exit 1
  fi

  logfile="$PWD/${PBS_LOG_DIR}/${stage}.log"
  if [[ -z "$prev_job" ]]; then
    jid=$(qsub -V -v PIPELINE_RUN_ID="$PIPELINE_RUN_ID" -j oe -o "$logfile" "$script")
  else
    jid=$(qsub -V -v PIPELINE_RUN_ID="$PIPELINE_RUN_ID" -j oe -o "$logfile" -W depend=afterok:${prev_job%%.*} "$script")
  fi
  echo "submitted ${stage}: ${jid}"
  prev_job="$jid"
done
