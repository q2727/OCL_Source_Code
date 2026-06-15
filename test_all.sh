#!/usr/bin/env bash
set -euo pipefail

RUNS="${RUNS:-30}"
SEED="${SEED:-25}"
METHODS="${METHODS:-kmd,ocl,wocl}"
WEIGHT_ALPHA="${WEIGHT_ALPHA:-0.5}"
WEIGHT_GAMMA="${WEIGHT_GAMMA:-1.0}"
WEIGHT_MIN="${WEIGHT_MIN:-auto}"
WEIGHT_DELAY="${WEIGHT_DELAY:-1}"
WEIGHT_MIX="${WEIGHT_MIX:-0.5}"
WEIGHT_GUARD="${WEIGHT_GUARD:-objective}"
WEIGHT_ENTROPY_MIN="${WEIGHT_ENTROPY_MIN:-0.7}"
WEIGHT_OBJECTIVE_TOL="${WEIGHT_OBJECTIVE_TOL:-0.0}"
CACHE_DIR="${CACHE_DIR:-.uv-cache}"
OUT_DIR="${OUT_DIR:-results/experiments}"
SCOPE="${SCOPE:-all}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "$OUT_DIR"

MAIN_OUT="$OUT_DIR/main_${TIMESTAMP}.md"
SUPP_OUT="$OUT_DIR/supplementary_${TIMESTAMP}.md"
SUMMARY_OUT="$OUT_DIR/all_${TIMESTAMP}.md"

run_experiment() {
  local title="$1"
  local data_root="$2"
  local output_file="$3"
  local -a datasets=()
  local dataset
  local total
  local idx

  {
    echo "# ${title}"
    echo
    echo "- date_utc: ${TIMESTAMP}"
    echo "- runs: ${RUNS}"
    echo "- seed: ${SEED}"
    echo "- methods: ${METHODS}"
    echo "- weight_alpha: ${WEIGHT_ALPHA}"
    echo "- weight_gamma: ${WEIGHT_GAMMA}"
    echo "- weight_min: ${WEIGHT_MIN}"
    echo "- weight_delay: ${WEIGHT_DELAY}"
    echo "- weight_mix: ${WEIGHT_MIX}"
    echo "- weight_guard: ${WEIGHT_GUARD}"
    echo "- weight_entropy_min: ${WEIGHT_ENTROPY_MIN}"
    echo "- weight_objective_tol: ${WEIGHT_OBJECTIVE_TOL}"
    echo "- data_root: ${data_root}"
    echo
    echo "Dataset | Method | CA | ARI | NMI | CMP"
    echo "--- | --- | --- | --- | --- | ---"
  } > "$output_file"

  mapfile -t datasets < <(
    find "$data_root" -maxdepth 1 -type f -name '*.mat' -printf '%f\n' \
      | sed 's/\.mat$//' \
      | sort
  )
  total="${#datasets[@]}"
  idx=0

  for dataset in "${datasets[@]}"; do
    idx=$((idx + 1))
    echo "  [$idx/$total] $dataset"
    uv run --cache-dir "$CACHE_DIR" ocl-run \
      --dataset "$dataset" \
      --data-root "$data_root" \
      --runs "$RUNS" \
      --seed "$SEED" \
      --methods "$METHODS" \
      --weight-alpha "$WEIGHT_ALPHA" \
      --weight-gamma "$WEIGHT_GAMMA" \
      --weight-min "$WEIGHT_MIN" \
      --weight-delay "$WEIGHT_DELAY" \
      --weight-mix "$WEIGHT_MIX" \
      --weight-guard "$WEIGHT_GUARD" \
      --weight-entropy-min "$WEIGHT_ENTROPY_MIN" \
      --weight-objective-tol "$WEIGHT_OBJECTIVE_TOL" \
      | sed '/^$/d; /^Dataset | Method |/d; /^--- | --- |/d' \
      >> "$output_file"
  done
}

if [[ "$SCOPE" != "all" && "$SCOPE" != "main" && "$SCOPE" != "supplementary" ]]; then
  echo "SCOPE must be one of: all, main, supplementary" >&2
  exit 2
fi

if [[ "$SCOPE" == "all" || "$SCOPE" == "main" ]]; then
  echo "Running main datasets: ${METHODS}, runs=${RUNS}, seed=${SEED}"
  run_experiment "Main Datasets: KMD vs OCL vs WOCL" "OCL_Source_Code/Data" "$MAIN_OUT"
  echo "Saved: $MAIN_OUT"
fi

if [[ "$SCOPE" == "all" || "$SCOPE" == "supplementary" ]]; then
  echo "Running supplementary datasets: ${METHODS}, runs=${RUNS}, seed=${SEED}"
  run_experiment "Supplementary Datasets: KMD vs OCL vs WOCL" "OCL_Source_Code/Supp_data" "$SUPP_OUT"
  echo "Saved: $SUPP_OUT"
fi

{
  echo "# All Experiments"
  echo
  echo "- date_utc: ${TIMESTAMP}"
  echo "- runs: ${RUNS}"
  echo "- seed: ${SEED}"
  echo "- methods: ${METHODS}"
  echo "- scope: ${SCOPE}"
  if [[ -f "$MAIN_OUT" ]]; then
    echo "- main_result: ${MAIN_OUT}"
  fi
  if [[ -f "$SUPP_OUT" ]]; then
    echo "- supplementary_result: ${SUPP_OUT}"
  fi
  echo
  if [[ -f "$MAIN_OUT" ]]; then
    cat "$MAIN_OUT"
    echo
    echo
  fi
  if [[ -f "$SUPP_OUT" ]]; then
    cat "$SUPP_OUT"
  fi
} > "$SUMMARY_OUT"

echo "Saved combined report: $SUMMARY_OUT"
