#!/usr/bin/env bash
set -u

rounds="${1:-10}"
run_offset="${RUN_OFFSET:-0}"
cd /Users/chrixbell/Projects/ValueInvestor
export PYTHONPYCACHEPREFIX=/tmp/valueinvestor_pycache
export PYTHONPATH=src

stamp() {
  date "+%Y-%m-%dT%H:%M:%S%z"
}

run=0
rows=(0)
refit_caps=(0)
limits=(1)
models=(
  market-ridge-recent-1095
  market-ridge-recent-730
  segment-ridge-recent-730
  segment-ridge-recent-730
  market-ridge-recent-1460
  market-ridge-recent-1825
  ridge-only
  market-ridge-recent-730
  market-ridge-recent-1095
  market-ridge-recent-1095
)
folds=(4)
presets=(
  "cross_sectional:rolling-ticker-priors:target-rank-weighted-901:1000"
  "cross_sectional:rolling-ticker-priors:target-rank-weighted-901:3000"
  "cross_sectional_interactions:rolling-ticker-priors:target-rank-weighted-802:0.25"
  "cross_sectional_interactions:rolling-ticker-priors:target-rank-weighted-8515-market:0.3"
  "cross_sectional:rolling-ticker-priors:target-rank-weighted-901:1000"
  "cross_sectional:rolling-ticker-priors:target-rank-weighted-901:1000"
  "core:rolling-ticker-priors:target-rank-6m:100"
  "cross_sectional_interactions:rolling-ticker-priors:target-rank-6m-soft:1000"
  "cross_sectional:rolling-ticker-priors:target-rank-weighted-703:1000"
  "cross_sectional_interactions:rolling-ticker-priors:target-rank-weighted-901:1000"
)
requested_backend="${BACKEND:-mlx}"
backend="$requested_backend"
mlx_auto_fallback="${MLX_AUTO_FALLBACK:-1}"
if [ "$mlx_auto_fallback" != "0" ]; then
  case "$backend" in
    mlx|mlx-cg|mlx-adam)
      if ! .venv/bin/python - <<'PY' >/dev/null 2>&1
import mlx.core as mx

x = mx.array([1.0])
mx.eval(x)
float(mx.sum(x).item())
PY
      then
        printf "[%s] requested backend=%s but MLX/Metal probe failed; using backend=numpy\n" \
          "$(stamp)" "$backend"
        backend=numpy
      fi
      ;;
  esac
fi
force_snapshots_on_first_run="${FORCE_SNAPSHOTS_ON_FIRST_RUN:-0}"
end_date="${END_DATE:-auto}"
gate_holdout_months="${GATE_HOLDOUT_MONTHS:-24}"
gate_embargo_days="${GATE_EMBARGO_DAYS:-197}"
gate_min_delta="${GATE_MIN_DELTA:-0.00002}"
gate_max_degradation="${GATE_MAX_DEGRADATION:-0.001}"
gate_min_weighted_utility="${GATE_MIN_WEIGHTED_UTILITY:-0.0}"
promotion_min_train_rho="${PROMOTION_MIN_TRAIN_RHO:-}"
promotion_min_gate_rho="${PROMOTION_MIN_GATE_RHO:-}"
auto_raise_promotion_min_train_rho="${AUTO_RAISE_PROMOTION_MIN_TRAIN_RHO:-0}"
export VALUEINVESTOR_ML_6M_GATE_MARKET_BLEND_LIMIT="${VALUEINVESTOR_ML_6M_GATE_MARKET_BLEND_LIMIT:-0}"
export VALUEINVESTOR_ML_6M_GATE_ROUTE_RAW_DEGRADATION_LIMIT="${VALUEINVESTOR_ML_6M_GATE_ROUTE_RAW_DEGRADATION_LIMIT:-0.005}"
export VALUEINVESTOR_ML_6M_RECENCY_HALF_LIVES_DAYS="${VALUEINVESTOR_ML_6M_RECENCY_HALF_LIVES_DAYS:-730,1095,1460,1825,2190}"
export VALUEINVESTOR_ML_6M_RECENT_SEGMENT_ROUTE_WEIGHTS="${VALUEINVESTOR_ML_6M_RECENT_SEGMENT_ROUTE_WEIGHTS:-1,0.5}"
export VALUEINVESTOR_ML_6M_RECENT_SEGMENT_ROUTE_MAX_STARTS="${VALUEINVESTOR_ML_6M_RECENT_SEGMENT_ROUTE_MAX_STARTS:-1}"
export VALUEINVESTOR_ML_6M_RECENT_SEGMENT_ROUTE_MAX_SEGMENTS="${VALUEINVESTOR_ML_6M_RECENT_SEGMENT_ROUTE_MAX_SEGMENTS:-2}"
export VALUEINVESTOR_ML_6M_RECENT_SEGMENT_ROUTE_MAX_CANDIDATES="${VALUEINVESTOR_ML_6M_RECENT_SEGMENT_ROUTE_MAX_CANDIDATES:-2}"
requested_sample_seeds="${VALUEINVESTOR_ML_6M_SAMPLE_SEEDS:-}"
export VALUEINVESTOR_ML_6M_SAMPLE_TRAIN_FLOOR_TOLERANCE="${VALUEINVESTOR_ML_6M_SAMPLE_TRAIN_FLOOR_TOLERANCE:-0.002}"
requested_refit_cap="${VALUEINVESTOR_ML_6M_FULL_EVAL_REFIT_MAX_ROWS:-}"
walk_forward_min_delta="${WALK_FORWARD_MIN_DELTA:-0.0}"
walk_forward_max_rows="${WALK_FORWARD_MAX_ROWS:-0}"
walk_forward_validation_months="${WALK_FORWARD_VALIDATION_MONTHS:-6}"
require_top20_excess="${REQUIRE_TOP20_EXCESS_NON_DEGRADATION:-1}"
anchor_model_paths="${ANCHOR_MODEL_PATHS:-none}"
goal_train_rho="${GOAL_TRAIN_RHO:-}"
goal_gate_rho="${GOAL_GATE_RHO:-}"
early_stop_on_goal="${EARLY_STOP_ON_GOAL:-1}"
output_model_path="${OUTPUT_MODEL_PATH:-data/trainer/ml_ranker_model.json}"
export OUTPUT_MODEL_PATH="$output_model_path"

printf "[%s] starting %s-round 6m ML scorer training loop\n" "$(stamp)" "$rounds"
printf "cwd=%s\n" "$PWD"
printf "run_offset=%s\n" "$run_offset"
printf "output_model_path=%s\n" "$output_model_path"
if [ -n "$goal_train_rho" ] || [ -n "$goal_gate_rho" ]; then
  printf "goal_train_rho=%s goal_gate_rho=%s\n" "${goal_train_rho:-none}" "${goal_gate_rho:-none}"
fi
printf "current_best_before=\n"
.venv/bin/python - <<'PY' || true
import json
import os
from pathlib import Path
from valueinvestor.scorer_improver.ml_trainer import _payload_cached_metrics_are_safe

p = Path(os.environ["OUTPUT_MODEL_PATH"])
payload = json.loads(p.read_text()) if p.exists() else {}
m = payload.get("metadata", {}) if isinstance(payload, dict) else {}
metrics_safe = isinstance(payload, dict) and _payload_cached_metrics_are_safe(payload)
train_metrics = m.get("train_metrics") or {}
gate_metrics = m.get("metrics") or {}
print({
    "backend": m.get("backend"),
    "model_kind": m.get("model_kind"),
    "feature_set": m.get("feature_set"),
    "target": m.get("target"),
    "prior_strategy": m.get("prior_strategy"),
    "cached_metrics_safe": metrics_safe,
    "train_rho_6m": (train_metrics.get("6m") or {}).get("spearman_rho") if metrics_safe else None,
    "gate_rho_6m": (gate_metrics.get("6m") or {}).get("spearman_rho") if metrics_safe else None,
})
PY

goal_met() {
  GOAL_TRAIN_RHO="$goal_train_rho" GOAL_GATE_RHO="$goal_gate_rho" .venv/bin/python - <<'PY'
import json
import os
from pathlib import Path
from valueinvestor.scorer_improver.ml_trainer import _payload_cached_metrics_are_safe

train_goal = os.environ.get("GOAL_TRAIN_RHO")
gate_goal = os.environ.get("GOAL_GATE_RHO")
if not train_goal and not gate_goal:
    raise SystemExit(1)

p = Path(os.environ["OUTPUT_MODEL_PATH"])
payload = json.loads(p.read_text()) if p.exists() else {}
m = payload.get("metadata", {}) if isinstance(payload, dict) else {}
metrics_safe = isinstance(payload, dict) and _payload_cached_metrics_are_safe(payload)
train_rho = ((m.get("train_metrics") or {}).get("6m") or {}).get("spearman_rho") if metrics_safe else None
gate_rho = ((m.get("metrics") or {}).get("6m") or {}).get("spearman_rho") if metrics_safe else None
train_ok = True if not train_goal else train_rho is not None and float(train_rho) >= float(train_goal)
gate_ok = True if not gate_goal else gate_rho is not None and float(gate_rho) >= float(gate_goal)
print({
    "train_rho_6m": train_rho,
    "gate_rho_6m": gate_rho,
    "cached_metrics_safe": metrics_safe,
    "goal_train_rho": train_goal or None,
    "goal_gate_rho": gate_goal or None,
    "goal_met": bool(train_ok and gate_ok),
})
raise SystemExit(0 if train_ok and gate_ok else 1)
PY
}

current_train_rho() {
  .venv/bin/python - <<'PY'
import json
import os
from pathlib import Path
from valueinvestor.scorer_improver.ml_trainer import _payload_cached_metrics_are_safe

p = Path(os.environ["OUTPUT_MODEL_PATH"])
payload = json.loads(p.read_text()) if p.exists() else {}
if not isinstance(payload, dict) or not _payload_cached_metrics_are_safe(payload):
    raise SystemExit(0)
m = payload.get("metadata", {})
rho = ((m.get("train_metrics") or {}).get("6m") or {}).get("spearman_rho")
if rho is not None:
    print(float(rho))
PY
}

should_raise_train_floor() {
  OLD_PROMOTION_MIN_TRAIN_RHO="$promotion_min_train_rho" \
  NEW_PROMOTION_MIN_TRAIN_RHO="$1" \
    .venv/bin/python - <<'PY'
import os

old_raw = os.environ.get("OLD_PROMOTION_MIN_TRAIN_RHO", "")
new_raw = os.environ.get("NEW_PROMOTION_MIN_TRAIN_RHO", "")
if not new_raw:
    raise SystemExit(1)
if not old_raw:
    raise SystemExit(0)
raise SystemExit(0 if float(new_raw) > float(old_raw) else 1)
PY
}

while [ "$run" -lt "$rounds" ]; do
  schedule_index=$(( run + run_offset ))
  preset=${presets[$(( schedule_index % ${#presets[@]} ))]}
  row=${rows[$(( schedule_index % ${#rows[@]} ))]}
  if [ -n "$requested_refit_cap" ]; then
    refit_cap="$requested_refit_cap"
  else
    refit_cap=${refit_caps[$(( schedule_index % ${#refit_caps[@]} ))]}
  fi
  export VALUEINVESTOR_ML_6M_FULL_EVAL_REFIT_MAX_ROWS="$refit_cap"
  limit=${limits[$(( (schedule_index / (${#presets[@]} * ${#rows[@]})) % ${#limits[@]} ))]}
  model=${models[$(( schedule_index % ${#models[@]} ))]}
  fold=${folds[$(( schedule_index % ${#folds[@]} ))]}
  if [ -n "$requested_sample_seeds" ]; then
    sample_seeds="$requested_sample_seeds"
  else
    sample_seeds="$schedule_index"
  fi
  export VALUEINVESTOR_ML_6M_SAMPLE_SEEDS="$sample_seeds"
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

  printf "\n[%s] run=%d rows=%s refit_cap=%s limit=%s model=%s features=%s priors=%s targets=%s lambdas=%s folds=%s seeds=%s backend=%s holdout_months=%s embargo_days=%s gate_min_delta=%s gate_max_degradation=%s gate_min_utility=%s promotion_min_train_rho=%s promotion_min_gate_rho=%s top20_excess_guard=%s anchors=%s force_snapshots=%s\n" \
    "$(stamp)" "$run" "$row" "$refit_cap" "$limit" "$model" "$feature_set" "$prior_set" "$target_set" \
    "$lambda_set" "$fold" "$sample_seeds" "$backend" "$gate_holdout_months" "$gate_embargo_days" \
    "$gate_min_delta" "$gate_max_degradation" "$gate_min_weighted_utility" \
    "${promotion_min_train_rho:-none}" "${promotion_min_gate_rho:-none}" "$require_top20_excess" "$anchor_model_paths" "$force_snapshots"

  set -- --target-horizon 6m --end-date "$end_date" --output-model-path "$output_model_path"
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
    --gate-holdout-months "$gate_holdout_months" \
    --gate-embargo-days "$gate_embargo_days" \
    --gate-min-6m-delta "$gate_min_delta" \
    --gate-max-degradation "$gate_max_degradation" \
    --gate-min-weighted-utility "$gate_min_weighted_utility" \
    --walk-forward-folds "$fold" \
    --walk-forward-min-6m-delta "$walk_forward_min_delta" \
    --walk-forward-max-rows "$walk_forward_max_rows" \
    --walk-forward-validation-months "$walk_forward_validation_months" \
    --full-eval-candidate-limit "$limit" \
    --anchor-model-paths "$anchor_model_paths" \
    --strict-outer-gate

  if [ -n "$promotion_min_train_rho" ]; then
    set -- "$@" --promotion-min-train-rho "$promotion_min_train_rho"
  fi
  if [ -n "$promotion_min_gate_rho" ]; then
    set -- "$@" --promotion-min-gate-rho "$promotion_min_gate_rho"
  fi

  case "$require_top20_excess" in
    0|false|FALSE|no|NO)
      set -- "$@" --gate-allow-top20-excess-degradation
      ;;
    *)
      set -- "$@" --gate-require-top20-excess-non-degradation
      ;;
  esac

  .venv/bin/python -m valueinvestor.cli.main train-ml-scorer "$@"
  status=$?
  printf "[%s] run=%d exit=%d\n" "$(stamp)" "$run" "$status"
  if [ "$status" -eq 130 ]; then
    printf "[%s] interrupted; stopping 6m ML scorer training loop\n" "$(stamp)"
    break
  fi
  if [ "$auto_raise_promotion_min_train_rho" != "0" ]; then
    latest_train_rho="$(current_train_rho || true)"
    if should_raise_train_floor "$latest_train_rho"; then
      promotion_min_train_rho="$latest_train_rho"
      printf "[%s] raised promotion_min_train_rho=%s\n" "$(stamp)" "$promotion_min_train_rho"
    fi
  fi
  if [ "$early_stop_on_goal" != "0" ] && { [ -n "$goal_train_rho" ] || [ -n "$goal_gate_rho" ]; }; then
    if goal_met; then
      printf "[%s] goal thresholds met after run=%d\n" "$(stamp)" "$run"
      break
    fi
  fi
done

printf "\n[%s] finished %s-round 6m ML scorer training loop\n" "$(stamp)" "$rounds"
printf "current_best_after=\n"
.venv/bin/python - <<'PY' || true
import json
import os
from pathlib import Path
from valueinvestor.scorer_improver.ml_trainer import _payload_cached_metrics_are_safe

p = Path(os.environ["OUTPUT_MODEL_PATH"])
payload = json.loads(p.read_text()) if p.exists() else {}
m = payload.get("metadata", {}) if isinstance(payload, dict) else {}
metrics_safe = isinstance(payload, dict) and _payload_cached_metrics_are_safe(payload)
train_metrics = m.get("train_metrics") or {}
gate_metrics = m.get("metrics") or {}
print({
    "backend": m.get("backend"),
    "model_kind": m.get("model_kind"),
    "feature_set": m.get("feature_set"),
    "target": m.get("target"),
    "prior_strategy": m.get("prior_strategy"),
    "cached_metrics_safe": metrics_safe,
    "train_rho_1w": (train_metrics.get("1w") or {}).get("spearman_rho") if metrics_safe else None,
    "train_rho_1m": (train_metrics.get("1m") or {}).get("spearman_rho") if metrics_safe else None,
    "train_rho_3m": (train_metrics.get("3m") or {}).get("spearman_rho") if metrics_safe else None,
    "train_rho_6m": (train_metrics.get("6m") or {}).get("spearman_rho") if metrics_safe else None,
    "gate_rho_6m": (gate_metrics.get("6m") or {}).get("spearman_rho") if metrics_safe else None,
})
PY
