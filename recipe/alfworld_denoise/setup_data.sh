#!/usr/bin/env bash
#
# Prepare offline ALFWorld + verl_agent data for the alfworld_denoise recipe.
#
# One-shot workflow on a machine with internet:
#
#   bash recipe/alfworld_denoise/setup_data.sh --download --offline
#
# That will:
#   1) download json_2.1.1_{json,pddl,tw-pddl}.zip into local_data/downloads/
#      using the githubfast.com mirror (configurable via --mirror);
#   2) extract them into local_data/alfworld/json_2.1.1/;
#   3) generate local_data/verl_agent/text/{train,test}.parquet via
#      examples.data_preprocess.prepare. Text-mode parquet rows are generated
#      locally and do not require Hugging Face data or cache access.
#
# On a no-network cluster, rsync local_data/ up first, then run with
# `--skip-extract` (or no flags at all -- it just verifies).
#
# Usage:
#   bash recipe/alfworld_denoise/setup_data.sh [options]
#
# Options:
#   --download            Download missing zips from the GitHub mirror.
#   --force-download      Re-download zips even if they already exist.
#   --no-download         Never download; fail if zips are missing (default
#                         behaviour when --download is not given).
#   --mirror URL          GitHub mirror base (default: https://githubfast.com).
#                         Other working values: https://github.com (slow from CN).
#   --release TAG         alfworld release tag (default: 0.2.2).
#   --skip-extract        Skip extracting the alfworld zips.
#   --skip-parquet        Skip (re)generating verl_agent parquet files.
#   --force-parquet       Regenerate parquet even if it already exists.
#   --offline             Force offline mode (text parquet is always local).
#   --train-size N        Train parquet row count (default: 16, matches params.sh).
#   --val-size N          Val/test parquet row count (default: 140, covers both val splits).
#   -h, --help            Show this help.
#
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RECIPE_DIR="${SCRIPT_DIR}"
REPO_ROOT=$(cd "${RECIPE_DIR}/../.." && pwd)

LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-${RECIPE_DIR}/local_data}"
DOWNLOADS_DIR="${DOWNLOADS_DIR:-${LOCAL_DATA_DIR}/downloads}"
ALFWORLD_DIR="${ALFWORLD_DIR:-${LOCAL_DATA_DIR}/alfworld}"
VERL_AGENT_DIR="${VERL_AGENT_DIR:-${LOCAL_DATA_DIR}/verl_agent}"

# githubfast.com is a transparent GitHub proxy that is much faster than
# github.com from inside China. The mirror is only contacted when --download
# is given. Override with --mirror or DOWNLOAD_MIRROR.
DEFAULT_MIRROR="https://githubfast.com"
DOWNLOAD_MIRROR="${DOWNLOAD_MIRROR:-${DEFAULT_MIRROR}}"
ALFWORLD_RELEASE="${ALFWORLD_RELEASE:-0.2.2}"

DOWNLOAD=0          # explicit --download
FORCE_DOWNLOAD=0    # --force-download
SKIP_EXTRACT=0
SKIP_PARQUET=0
FORCE_PARQUET=0
OFFLINE=0
TRAIN_SIZE=${TRAIN_BATCH_SIZE:-16}
VAL_SIZE=${VAL_DATA_SIZE:-140}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --download)         DOWNLOAD=1; shift ;;
    --force-download)   DOWNLOAD=1; FORCE_DOWNLOAD=1; shift ;;
    --no-download)      DOWNLOAD=0; FORCE_DOWNLOAD=0; shift ;;
    --mirror)           DOWNLOAD_MIRROR="$2"; shift 2 ;;
    --release)          ALFWORLD_RELEASE="$2"; shift 2 ;;
    --skip-extract)     SKIP_EXTRACT=1; shift ;;
    --skip-parquet)     SKIP_PARQUET=1; shift ;;
    --force-parquet)    FORCE_PARQUET=1; shift ;;
    --offline)          OFFLINE=1; shift ;;
    --train-size)       TRAIN_SIZE="$2"; shift 2 ;;
    --val-size)         VAL_SIZE="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

echo "============================================================"
echo " ALFWorld DenoiseRL offline data setup"
echo "   LOCAL_DATA_DIR : ${LOCAL_DATA_DIR}"
echo "   DOWNLOADS_DIR  : ${DOWNLOADS_DIR}"
echo "   ALFWORLD_DIR   : ${ALFWORLD_DIR}"
echo "   VERL_AGENT_DIR : ${VERL_AGENT_DIR}"
echo "   download       : ${DOWNLOAD} (force=${FORCE_DOWNLOAD})"
echo "   mirror         : ${DOWNLOAD_MIRROR}"
echo "   release        : ${ALFWORLD_RELEASE}"
echo "   skip-extract   : ${SKIP_EXTRACT}"
echo "   skip-parquet   : ${SKIP_PARQUET}"
echo "   force-parquet  : ${FORCE_PARQUET}"
echo "   offline        : ${OFFLINE}"
echo "   train/val size : ${TRAIN_SIZE} / ${VAL_SIZE}"
echo "============================================================"

mkdir -p "${LOCAL_DATA_DIR}" "${DOWNLOADS_DIR}" "${ALFWORLD_DIR}" "${VERL_AGENT_DIR}"

# -----------------------------------------------------------------------------
# Step 0: download ALFWorld json_2.1.1 zips from the GitHub mirror
# -----------------------------------------------------------------------------
EXPECTED_ZIPS=(
  json_2.1.1_json.zip
  json_2.1.1_pddl.zip
  json_2.1.1_tw-pddl.zip
)
# Rough lower bounds (bytes). Used to detect truncated/HTML error pages.
# (Plain case statement so this works on macOS bash 3.2 as well.)
_min_size_for() {
  case "$1" in
    json_2.1.1_json.zip)    echo 60000000 ;;  # ~69 MB
    json_2.1.1_pddl.zip)    echo 28000000 ;;  # ~33 MB
    json_2.1.1_tw-pddl.zip) echo 38000000 ;;  # ~43 MB
    *)                      echo 0 ;;
  esac
}

# Returns 0 if ${DOWNLOADS_DIR}/${1} exists and is larger than its min size.
zip_ok() {
  local name="$1" path="${DOWNLOADS_DIR}/$1"
  [[ -f "${path}" ]] || return 1
  local sz min
  sz=$(stat -c%s "${path}" 2>/dev/null || stat -f%z "${path}" 2>/dev/null || echo 0)
  min=$(_min_size_for "${name}")
  [[ "${sz}" -ge "${min}" ]]
}

if [[ "${DOWNLOAD}" -eq 1 ]]; then
  for z in "${EXPECTED_ZIPS[@]}"; do
    out="${DOWNLOADS_DIR}/${z}"
    if zip_ok "${z}" && [[ "${FORCE_DOWNLOAD}" -eq 0 ]]; then
      echo "[download] ok: ${out} (already present)"
      continue
    fi
    url="${DOWNLOAD_MIRROR%/}/alfworld/alfworld/releases/download/${ALFWORLD_RELEASE}/${z}"
    echo "[download] ${url} -> ${out}"
    # -L: follow redirects (githubfast -> github release assets).
    # -C -: resume partial downloads so re-runs are cheap.
    # -f: fail silently on server errors so we don't save a 404 HTML page.
    if ! curl -L -f -C - -o "${out}" "${url}"; then
      echo "[download] ERROR: failed to fetch ${url}" >&2
      echo "   try a different mirror, e.g. --mirror https://github.com" >&2
      exit 1
    fi
    if ! zip_ok "${z}"; then
      echo "[download] ERROR: ${out} is too small or corrupt; delete it and re-run." >&2
      exit 1
    fi
  done
fi

# -----------------------------------------------------------------------------
# Step 1: logic files (alfred.pddl / alfred.twl2)
#
# These are tiny (~30 KB total) and shipped with the alfworld env_package in
# the repo. The recipe commit already placed copies under
#   recipe/alfworld_denoise/local_data/alfworld/logic/
# so on the remote they will arrive via the rsync. We only (re)create them if
# missing.
# -----------------------------------------------------------------------------
LOGIC_DIR="${ALFWORLD_DIR}/logic"
mkdir -p "${LOGIC_DIR}"
BUILTIN_LOGIC="${REPO_ROOT}/agent_system/environments/env_package/alfworld/alfworld/data"
for f in alfred.pddl alfred.twl2; do
  if [[ ! -f "${LOGIC_DIR}/${f}" ]]; then
    if [[ -f "${BUILTIN_LOGIC}/${f}" ]]; then
      cp "${BUILTIN_LOGIC}/${f}" "${LOGIC_DIR}/${f}"
      echo "[logic] copied ${f} from env_package"
    else
      echo "[logic] WARNING: ${LOGIC_DIR}/${f} missing and no builtin source at ${BUILTIN_LOGIC}/${f}" >&2
    fi
  else
    echo "[logic] ok: ${LOGIC_DIR}/${f}"
  fi
done

# -----------------------------------------------------------------------------
# Step 2: extract ALFWorld json_2.1.1 zips
# -----------------------------------------------------------------------------
if [[ "${SKIP_EXTRACT}" -eq 1 ]]; then
  echo "[extract] skipped (--skip-extract)"
else
  missing=()
  for z in "${EXPECTED_ZIPS[@]}"; do
    if ! zip_ok "${z}"; then
      missing+=("${z}")
    fi
  done
  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "[extract] ERROR: missing or truncated zips under ${DOWNLOADS_DIR}:" >&2
    printf '   - %s\n' "${missing[@]}" >&2
    if [[ "${DOWNLOAD}" -eq 0 ]]; then
      echo "   Re-run with --download to fetch them automatically from ${DOWNLOAD_MIRROR}," >&2
      echo "   or fetch manually:" >&2
      echo "     cd ${DOWNLOADS_DIR}" >&2
      echo "     for z in ${EXPECTED_ZIPS[*]}; do" >&2
      echo "       curl -L -f -C - -o \"\$z\" \"${DOWNLOAD_MIRROR%/}/alfworld/alfworld/releases/download/${ALFWORLD_RELEASE}/\$z\"" >&2
      echo "     done" >&2
    fi
    exit 1
  fi

  echo "[extract] extracting into ${ALFWORLD_DIR} ..."
  for z in "${EXPECTED_ZIPS[@]}"; do
    echo "   - ${z}"
    unzip -q -o "${DOWNLOADS_DIR}/${z}" -d "${ALFWORLD_DIR}"
  done
  echo "[extract] done."
fi

# Sanity-check the expected layout.
for split in train valid_seen valid_unseen; do
  if [[ ! -d "${ALFWORLD_DIR}/json_2.1.1/${split}" ]]; then
    echo "[verify] WARNING: ${ALFWORLD_DIR}/json_2.1.1/${split} not found." >&2
  else
    n=$(find "${ALFWORLD_DIR}/json_2.1.1/${split}" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
    echo "[verify] json_2.1.1/${split}: ${n} game(s)"
  fi
done

# -----------------------------------------------------------------------------
# Step 3: verl_agent parquet (train/test)
# -----------------------------------------------------------------------------
TRAIN_PARQUET="${VERL_AGENT_DIR}/text/train.parquet"
TEST_PARQUET="${VERL_AGENT_DIR}/text/test.parquet"

parquet_row_count() {
  python3 -c 'import sys; import pyarrow.parquet as pq; print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)' "$1"
}

parquet_has_required_rows() {
  [[ -f "${TRAIN_PARQUET}" && -f "${TEST_PARQUET}" ]] || return 1
  local train_rows val_rows
  train_rows=$(parquet_row_count "${TRAIN_PARQUET}") || return 1
  val_rows=$(parquet_row_count "${TEST_PARQUET}") || return 1
  [[ "${train_rows}" -ge "${TRAIN_SIZE}" && "${val_rows}" -ge "${VAL_SIZE}" ]]
}

if [[ "${SKIP_PARQUET}" -eq 1 ]]; then
  echo "[parquet] skipped (--skip-parquet)"
elif [[ "${FORCE_PARQUET}" -eq 0 ]] && parquet_has_required_rows; then
  echo "[parquet] already exists, skipping (use --force-parquet to regenerate):"
  echo "   - ${TRAIN_PARQUET}"
  echo "   - ${TEST_PARQUET}"
else
  if [[ -f "${TRAIN_PARQUET}" || -f "${TEST_PARQUET}" ]]; then
    echo "[parquet] existing files are missing rows or unreadable; regenerating."
    echo "   required rows: train>=${TRAIN_SIZE}, validation>=${VAL_SIZE}"
  fi
  mkdir -p "${VERL_AGENT_DIR}/text"
  echo "[parquet] generating via examples.data_preprocess.prepare ..."
  EXTRA_ENV=()
  EXTRA_ARGS=()
  if [[ "${OFFLINE}" -eq 1 ]]; then
    EXTRA_ENV+=(env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1)
    EXTRA_ARGS+=(--offline)
  fi
  (
    cd "${REPO_ROOT}"
    "${EXTRA_ENV[@]}" python3 -m examples.data_preprocess.prepare \
      --mode text \
      --local_dir "${VERL_AGENT_DIR}" \
      --train_data_size "${TRAIN_SIZE}" \
      --val_data_size "${VAL_SIZE}" \
      "${EXTRA_ARGS[@]}"
  )
  echo "[parquet] done."
fi

# -----------------------------------------------------------------------------
# Final summary
# -----------------------------------------------------------------------------
echo "============================================================"
echo " Done. Layout:"
echo "   ${ALFWORLD_DIR}/json_2.1.1/{train,valid_seen,valid_unseen}"
echo "   ${ALFWORLD_DIR}/logic/{alfred.pddl,alfred.twl2}"
echo "   ${VERL_AGENT_DIR}/text/{train,test}.parquet"
echo
echo " To train, point the scripts at this layout by exporting:"
echo "   export ALFWORLD_DATA=\"${ALFWORLD_DIR}\""
echo "   export OFFLINE_DATA_ONLY=1"
echo "============================================================"
