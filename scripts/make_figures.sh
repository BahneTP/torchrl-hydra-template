#!/usr/bin/env bash
#
# Comparison figures + rliable aggregates, straight from W&B.
#
# This is a thin wrapper around openrlbenchmark's own `rlops` CLI
# (https://github.com/openrlbenchmark/openrlbenchmark) — no plotting code of our
# own. It works without adapters because the logging contract already matches
# what rlops expects: top-level `env_id` / `exp_name` / `seed`, one row per
# episode on `charts/episodic_return`, and `global_step` as a real column.
# `tests/test_evaluation_contract.py` is what keeps that true.
#
#   ./scripts/make_figures.sh                    # every group, tag `template`
#   ./scripts/make_figures.sh --group atari100k  # one group
#   ./scripts/make_figures.sh --tag eval-recheck # a different W&B tag
#   ./scripts/make_figures.sh --entity LatentLab --project torchrl-hydra-template
#
# Output: logs/analysis/<group>{,_aggregate,_performance_profile,
# _sample_efficiency}.{png,pdf,svg} plus a markdown/csv score table.
#
# READ THE CAVEAT BEFORE USING THE NUMBERS: rlops averages the last
# `--metric-last-n-average-window` (100) logged points of
# `charts/episodic_return`, whatever stream `canonical_source` pointed it at.
# Runs recorded under different evaluation protocols are therefore NOT
# comparable, and rlops cannot detect that. Two failure modes seen in practice:
#
#   * a run whose canonical episodes all sit at one step (periodic eval off)
#     renders as a flat horizontal line across the whole x-axis, because the
#     single point is back-filled — it looks like a curve and is not one;
#   * a run that crashed early still shows as `finished` on W&B (the trainer
#     closes the logger in a `finally`), so a truncated run silently joins the
#     comparison. Check the seed/step counts rlops prints before trusting a plot.
#
set -uo pipefail
# `?` and `&` in the filter strings must reach rlops literally.
set -f

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

ENTITY="${WANDB_ENTITY:-LatentLab}"
PROJECT="${WANDB_PROJECT:-torchrl-hydra-template}"
TAG="template"
GROUP=""
OUT_DIR="logs/analysis"
VENV=".venv-openrlbenchmark"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --entity)   ENTITY="$2"; shift 2 ;;
    --project)  PROJECT="$2"; shift 2 ;;
    --tag)      TAG="$2"; shift 2 ;;
    --group)    GROUP="$2"; shift 2 ;;
    --out)      OUT_DIR="$2"; shift 2 ;;
    -h|--help)  sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

# ------------------------------------------------------------------ toolchain
# openrlbenchmark pulls seaborn / expt / older numpy pins. Installing it into
# the training venv risks moving torch's numpy out from under it, so it gets its
# own environment; `uv` is already a dependency of the devcontainer.
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "building $VENV (one-off)"
  uv venv --python 3.10 "$VENV" || exit 1
  uv pip install --python "$VENV/bin/python" --quiet openrlbenchmark || exit 1
fi

mkdir -p "$OUT_DIR"

# -------------------------------------------------------------------- groups
#
# Format: NAME | ENV-IDS | NORMALIZATION | EXPERIMENTS (exp_name:label,...)
#
# `exp_name` is the top-level config key (= the algorithm config's name), which
# is what rlops matches on via `cen=exp_name`. `atari` normalization uses
# openrlbenchmark's built-in human/random table, keyed `Jamesbond-v5` — the
# reason `env_id` carries no `ALE/` prefix.
# NOT `GROUPS`: bash reserves that name for the caller's group IDs and silently
# ignores the assignment, so every field would parse as a gid.
FIG_GROUPS=(
  "atari100k|Jamesbond-v5|atari|dreamer:DreamerV3,bbf:BBF,ppo:PPO"
  "dmc|cheetah-run|maxmin|dreamer:DreamerV3,tdmpc2:TD-MPC2,ppo:PPO"
)

status=0
for spec in "${FIG_GROUPS[@]}"; do
  name="${spec%%|*}";      rest="${spec#*|}"
  env_ids="${rest%%|*}";   rest="${rest#*|}"
  norm="${rest%%|*}";      experiments="${rest#*|}"

  [[ -n "$GROUP" && "$GROUP" != "$name" ]] && continue

  # 'dreamer:DreamerV3,bbf:BBF' -> 'dreamer?tag=template&cl=DreamerV3' ...
  args=()
  IFS=',' read -ra pairs <<< "$experiments"
  for pair in "${pairs[@]}"; do
    args+=("${pair%%:*}?tag=${TAG}&cl=${pair#*:}")
  done

  echo "=== $name  (env_ids=$env_ids  norm=$norm  tag=$TAG)"
  "$VENV/bin/python" -m openrlbenchmark.rlops \
    --filters "?we=${ENTITY}&wpn=${PROJECT}&ceik=env_id&cen=exp_name&metric=charts/episodic_return" \
      "${args[@]}" \
    --env-ids $env_ids \
    --no-check-empty-runs \
    --pc.ncols 1 --pc.ncols-legend 3 \
    --rliable \
    --rc.score-normalization-method "$norm" \
    --rc.normalized-score-threshold 8.0 \
    --rc.sample-efficiency-plots \
    --rc.performance-profile-plots \
    --rc.aggregate-metrics-plots \
    --output-filename "${OUT_DIR}/${name}" \
    --scan-history || status=1
done

echo
echo "figures written to ${OUT_DIR}/"
exit $status
