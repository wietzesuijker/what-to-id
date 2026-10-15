#!/usr/bin/env bash
# One daily build of the rotation page. GitHub Actions (.github/workflows/daily.yml) runs
# exactly these steps, and so can a laptop:
#
#   scripts/daily.sh fetch   download the pool and served log from the pool-state release
#   scripts/daily.sh build   load the prepared pool, build, drop identified records, rebuild,
#                            keep the day's pool and build record, check for leaks, replay
#   scripts/daily.sh save    upload the state back to the release
#
# build needs WHAT_TO_ID_KEY (the private list key) and WTB_DIR (where-to-blitz's
# cluster_results/ca, checked out at the commit in what_to_id.manifest.WHERE_TO_BLITZ_REF).
# OFFLINE=1 skips the two iNaturalist calls, to rehearse the rest on a saved pool.
# Nothing here may print the key or anything that maps list letters to lists.
set -euo pipefail
cd "$(dirname "$0")/.."

STATE=${STATE_DIR:-state}
OUT=${OUT_DIR:-out}
TODAY=${TODAY:-$(date -u +%F)}
D1=${D1:-2025-01-01}
STATE_TAG=${STATE_TAG:-pool-state}
MAX_BATCHES=${MAX_BATCHES:-20}
ARMS=${ARMS:-recency,gap_first,similarity,novelty}
EMBEDDING_BUNDLE=${EMBEDDING_BUNDLE:-data/embedding-bundle}
PY=${PYTHON:-python}
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

die() {
  echo "daily: $*" >&2
  exit 1
}

state_release_exists() {
  local response
  if response=$(gh api --include --silent "repos/$1/releases/tags/$STATE_TAG" 2>&1); then
    return 0
  fi
  if grep -Eq '^HTTP/[0-9.]+ 404([[:space:]]|$)' <<<"$response"; then
    return 1
  fi
  die "cannot read $1 release $STATE_TAG; refusing to assume this is the first build"
}

fetch() {
  local state_repo=${STATE_REPO:-${GITHUB_REPOSITORY:-PollockLab/what-to-id}}
  mkdir -p "$STATE"
  if ! state_release_exists "$state_repo"; then
    echo "no $STATE_TAG release yet: the first build pulls the whole pool"
    return
  fi
  local assets
  assets=$(gh release view "$STATE_TAG" --repo "$state_repo" --json assets --jq '.assets[].name')
  for f in pool.parquet served.parquet bundle-eligibility.json; do
    if grep -qx "$f" <<<"$assets"; then
      gh release download "$STATE_TAG" --repo "$state_repo" -D "$STATE" -p "$f" --clobber
    fi
  done
  if grep -q '^build-.*\.json$' <<<"$assets" && [ ! -s "$STATE/served.parquet" ]; then
    die "the release holds earlier builds but no served log; restore served.parquet before building"
  fi
  ls -la "$STATE"
}

fetch_embeddings() {
  : "${EMBEDDING_RELEASE:?set EMBEDDING_RELEASE to a versioned embedding bundle release tag}"
  local source_repo=${EMBEDDING_REPO:-${GITHUB_REPOSITORY:-PollockLab/what-to-id}}
  [ ! -e "$EMBEDDING_BUNDLE" ] || die "embedding destination exists; use a fresh EMBEDDING_BUNDLE path"
  local downloads="${EMBEDDING_BUNDLE}.download"
  [ ! -e "$downloads" ] || die "embedding download destination exists; use a fresh path"
  mkdir -p "$downloads"
  gh release download "$EMBEDDING_RELEASE" --repo "$source_repo" -D "$downloads" -p '*'
  $PY -m what_to_id.bundle_transport restore --directory "$downloads" --out "$EMBEDDING_BUNDLE"
  rm -rf "$downloads" # Verified parts can be fetched again; keep runner disk for the build.
}

embedding_args() {
  EMBEDDING_ARGS=()
  case ",$ARMS," in
    *,similarity,*|*,novelty,*)
      [ -s "$EMBEDDING_BUNDLE/embedding_bundle.json" ] || die "embedding lists require a prepared EMBEDDING_BUNDLE"
      $PY -m what_to_id.artifacts verify "$EMBEDDING_BUNDLE"
      EMBEDDING_ARGS=(--embedding-bundle "$EMBEDDING_BUNDLE")
      ;;
  esac
}

check_wtb() {
  [ -n "${WTB_DIR:-}" ] || die "set WTB_DIR to where-to-blitz's cluster_results/ca"
  [ -f "$WTB_DIR/provenance.json" ] || die "$WTB_DIR has no provenance.json"
  local want have
  want=$($PY -c "from what_to_id.manifest import WHERE_TO_BLITZ_REF as r; print(r.split('@')[1])")
  have=$(git -C "$WTB_DIR" rev-parse HEAD)
  [ "$want" = "$have" ] || die "where-to-blitz is at $have, the code expects $want"
}

check_leaks() {
  local site="$OUT/final/site"
  [ -s "$site/index.html" ] || die "no page was built"
  if find "$site" -type f ! -name '*.html' ! -name 'pool*.bin' ! -name bc-parks.bin | grep -q .; then
    die "the site must hold only HTML pages, the map's pool*.bin files and bc-parks.bin"
  fi
  if grep -ril -E 'recency|gap_first|gap first|similarity|novelty|surprise' "$site"; then
    die "a list name leaked into the site"
  fi
  if grep -rlF "$WHAT_TO_ID_KEY" "$site" "$STATE"; then
    die "the list key leaked into a file that is published or uploaded"
  fi
  $PY - "$STATE/served.parquet" "$STATE/days/build-$TODAY.json" <<'EOF'
import json, sys
import pandas as pd
served = pd.read_parquet(sys.argv[1])
assert "arm" not in served.columns, "the served log names lists"
record = json.load(open(sys.argv[2]))
for k in ("arm_labels", "batches"):
    assert k not in record, f"the build record carries {k}"
EOF
}

build() {
  [ -n "${WHAT_TO_ID_KEY:-}" ] || die "WHAT_TO_ID_KEY is not set"
  : "${BLITZ_D1:?set BLITZ_D1, the blitz start date (YYYY-MM-DD)}"
  check_wtb
  embedding_args

  mkdir -p "$STATE/days"
  local eligibility_args=()
  if [ "${#EMBEDDING_ARGS[@]}" -gt 0 ]; then
    [ "${FULL_PULL:-}" != true ] || die "prepare and publish a fresh bundle for a full pool pull"
    [ -s "$EMBEDDING_BUNDLE/pool.parquet" ] || die "bundle has no prepared pool; repack it"
    # Publish a complete snapshot and its embeddings together. Adding observations here
    # would race preparation and make their embeddings unavailable to this build.
    eligibility_args=(--eligibility "$STATE/bundle-eligibility.json")
    $PY -m what_to_id.pool_state prepared --pool "$STATE/pool.parquet" \
      --prepared-pool "$EMBEDDING_BUNDLE/pool.parquet" "${eligibility_args[@]}"
  elif [ "${OFFLINE:-}" = 1 ]; then
    [ -s "$STATE/pool.parquet" ] || die "OFFLINE=1 needs a saved $STATE/pool.parquet"
  elif [ "${FULL_PULL:-}" = true ] || [ ! -s "$STATE/pool.parquet" ]; then
    $PY -m what_to_id.inat --d1 "$D1" --freeze "$TODAY" --out "$STATE/pool.parquet"
  else
    $PY -m what_to_id.pool_state update --pool "$STATE/pool.parquet" --d1 "$D1"
  fi

  local common=(--pool "$STATE/pool.parquet" --freeze "$TODAY" --d1 "$BLITZ_D1"
    --arms "$ARMS" --design rotation --max-batches "$MAX_BATCHES" --key-env WHAT_TO_ID_KEY
    --webapp-dir "$WTB_DIR" "${EMBEDDING_ARGS[@]}")
  rm -rf "$OUT/draft" "$OUT/final" "$OUT/ordering-cache"
  mkdir -p "$OUT/ordering-cache"
  chmod 700 "$OUT/ordering-cache"
  $PY -m what_to_id.cli build "${common[@]}" --out "$OUT/draft" --ordering-cache "$OUT/ordering-cache"
  if [ "${OFFLINE:-}" != 1 ]; then
    $PY -m what_to_id.pool_state refresh --pool "$STATE/pool.parquet" --ids "$OUT/draft/batches.parquet" "${eligibility_args[@]}"
  fi
  $PY -m what_to_id.cli build "${common[@]}" --out "$OUT/final" --ordering-cache "$OUT/ordering-cache" --served-log "$STATE/served.parquet"

  # The day's exact inputs, kept so this build can be rerun later.
  cp "$STATE/pool.parquet" "$STATE/days/pool-$TODAY.parquet"
  cp "$OUT/final/build_record.json" "$STATE/days/build-$TODAY.json"
  check_leaks
  $PY -m what_to_id.replay --pool "$STATE/days/pool-$TODAY.parquet" \
    --record "$STATE/days/build-$TODAY.json" --served-log "$STATE/served.parquet" \
    --webapp-dir "$WTB_DIR" --key-env WHAT_TO_ID_KEY "${EMBEDDING_ARGS[@]}"
}

save() {
  local state_repo=${STATE_REPO:-${GITHUB_REPOSITORY:-PollockLab/what-to-id}}
  local files=("$STATE/pool.parquet" "$STATE/served.parquet"
    "$STATE/days/pool-$TODAY.parquet" "$STATE/days/build-$TODAY.json")
  if [ -f "$STATE/bundle-eligibility.json" ]; then
    files+=("$STATE/bundle-eligibility.json")
  fi
  for f in "${files[@]}"; do
    [ -s "$f" ] || die "missing $f; run build first"
  done
  if ! state_release_exists "$state_repo"; then
    gh release create "$STATE_TAG" --repo "$state_repo" \
      --prerelease --latest=false --title "Daily pool state" \
      --notes "Pool, served log (list letters only), and each day's pool and build record, kept between daily builds."
  fi
  local attempt
  for attempt in 1 2 3; do
    if gh release upload "$STATE_TAG" "${files[@]}" --repo "$state_repo" --clobber; then
      return
    fi
    echo "upload failed (attempt $attempt), retrying"
    sleep $((attempt * 30))
  done
  die "could not save the state to the $STATE_TAG release"
}

case "${1:-}" in
  fetch) fetch ;;
  fetch-embeddings) fetch_embeddings ;;
  build) build ;;
  save) save ;;
  *) die "usage: scripts/daily.sh fetch|fetch-embeddings|build|save" ;;
esac
