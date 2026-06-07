#!/usr/bin/env bash
set -u

hours="${1:-10}"
cd /Users/chrixbell/Projects/ValueInvestor
export PYTHONPYCACHEPREFIX=/tmp/valueinvestor_pycache
export PYTHONPATH=src

stamp() {
  date "+%Y-%m-%dT%H:%M:%S%z"
}

end=$(( $(date +%s) + hours * 3600 ))
run=0
rows=(1000000 750000 1500000 500000 2000000)
limits=(12 16)
models=(market-ridge ridge)
folds=(0 3)
presets=(
  "short-horizon:no-ticker-priors:target-rank-1w:0.3,0.5,1,2,3,5,10,30"
  "short-horizon:no-ticker-priors:target-rank-1w:100,300,1000"
  "short-horizon:no-ticker-priors:target-rank-1w-market:0.3,0.5,1,2,3,5,10,30"
)
backend=auto
force_snapshots_on_first_run="${FORCE_SNAPSHOTS_ON_FIRST_RUN:-1}"
gate_min_delta="${GATE_MIN_DELTA:-0.000001}"
gate_min_weighted_utility="${GATE_MIN_WEIGHTED_UTILITY:--0.0001}"
require_top20_excess="${REQUIRE_TOP20_EXCESS_NON_DEGRADATION:-0}"

printf "[%s] starting %sh 1w ML scorer training loop\n" "$(stamp)" "$hours"
printf "cwd=%s\n" "$PWD"
printf "current_best_before=\n"
.venv/bin/python - <<'PY' || true
import json
from pathlib import Path

p = Path("data/trainer/ml_ranker_model_1w.json")
m = json.loads(p.read_text()).get("metadata", {}) if p.exists() else {}
metrics = m.get("metrics") or {}
print({
    "backend": m.get("backend"),
    "model_kind": m.get("model_kind"),
    "feature_set": m.get("feature_set"),
    "target": m.get("target"),
    "prior_strategy": m.get("prior_strategy"),
    "rho_1w": (metrics.get("1w") or {}).get("spearman_rho"),
})
PY

while [ "$(date +%s)" -lt "$end" ]; do
  preset=${presets[$(( run % ${#presets[@]} ))]}
  row=${rows[$(( (run / ${#presets[@]}) % ${#rows[@]} ))]}
  limit=${limits[$(( (run / (${#presets[@]} * ${#rows[@]})) % ${#limits[@]} ))]}
  model=${models[$(( (run / (${#presets[@]} * ${#rows[@]} * ${#limits[@]})) % ${#models[@]} ))]}
  fold=${folds[$(( (run / (${#presets[@]} * ${#rows[@]} * ${#limits[@]} * ${#models[@]})) % ${#folds[@]} ))]}
  remaining="$preset"
  feature_set="${remaining%%:*}"
  remaining="${remaining#*:}"
  prior_set="${remaining%%:*}"
  remaining="${remaining#*:}"
  target_set="${remaining%%:*}"
  lambda_set="${remaining#*:}"
  force_snapshots=no
  if [ "$run" -eq 0 ] && [ "$force_snapshots_on_first_run" != "0" ]; then
    force_snapshots=yes
  fi
  run=$((run + 1))

  printf "\n[%s] run=%d rows=%s limit=%s model=%s features=%s priors=%s targets=%s lambdas=%s folds=%s backend=%s gate_min_delta=%s gate_min_utility=%s top20_excess_guard=%s force_snapshots=%s\n" \
    "$(stamp)" "$run" "$row" "$limit" "$model" "$feature_set" "$prior_set" "$target_set" \
    "$lambda_set" "$fold" "$backend" "$gate_min_delta" "$gate_min_weighted_utility" \
    "$require_top20_excess" "$force_snapshots"

  set -- --target-horizon 1w
  if [ "$force_snapshots" = "yes" ]; then
    set -- "$@" --force-snapshots
  fi
  set -- "$@" \
    --backend "$backend" \
    --model-kind "$model" \
    --candidate-feature-sets "$feature_set" \
    --candidate-prior-strategies "$prior_set" \
    --candidate-targets "$target_set" \
    --candidate-ridge-lambdas "$lambda_set" \
    --max-training-rows "$row" \
    --gate-min-6m-delta "$gate_min_delta" \
    --gate-min-weighted-utility "$gate_min_weighted_utility" \
    --walk-forward-folds "$fold" \
    --walk-forward-max-rows 500000 \
    --full-eval-candidate-limit "$limit"

  case "$require_top20_excess" in
    1|true|TRUE|yes|YES)
      set -- "$@" --gate-require-top20-excess-non-degradation
      ;;
    *)
      set -- "$@" --gate-allow-top20-excess-degradation
      ;;
  esac

  .venv/bin/python -m valueinvestor.cli.main train-ml-scorer "$@"

  status=$?
  printf "[%s] run=%d exit=%d\n" "$(stamp)" "$run" "$status"
  sleep 20
done

printf "\n[%s] finished %sh 1w ML scorer training loop after %d run(s)\n" \
  "$(stamp)" "$hours" "$run"
printf "current_best_after=\n"
.venv/bin/python - <<'PY' || true
import json
from pathlib import Path

p = Path("data/trainer/ml_ranker_model_1w.json")
m = json.loads(p.read_text()).get("metadata", {}) if p.exists() else {}
metrics = m.get("metrics") or {}
print({
    "backend": m.get("backend"),
    "model_kind": m.get("model_kind"),
    "feature_set": m.get("feature_set"),
    "target": m.get("target"),
    "prior_strategy": m.get("prior_strategy"),
    "rho_1w": (metrics.get("1w") or {}).get("spearman_rho"),
    "rho_1m": (metrics.get("1m") or {}).get("spearman_rho"),
    "rho_3m": (metrics.get("3m") or {}).get("spearman_rho"),
    "rho_6m": (metrics.get("6m") or {}).get("spearman_rho"),
})
PY
