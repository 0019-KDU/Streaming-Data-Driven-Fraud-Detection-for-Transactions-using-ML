# Postmortem: silent train/serve skew in the inference feature pipeline

**Status:** fixed (shared pipeline module and parity tests). The follow-up below removes the root causes; it needs a retrain to take effect.

## Summary

Every transaction scored through the streaming pipeline failed, and the dashboard showed nothing, for a period in which the services were "up" and logged no crash. The cause was train/serve skew of the most basic kind. The serving path no longer ran the feature-engineering code the model was trained with.

## Timeline

| When | What |
|---|---|
| Nov 2025 | Inference works. Spark processes a test transaction (`TransactionID 2987245`, $37.10) end to end: probability 0.019, `APPROVE`. One trigger took 17.8 s, almost all of it row-by-row feature engineering. |
| Commit `023e4d7` | `IEEECISFeaturePipeline` and its feature methods were removed from `feature_pipeline.py` during a refactor. |
| Commit `dcbf894` | `src/dags/feature_pipeline.py` was deleted. |
| Commit `2d5aef4` | "Fix broken DAG". A 168-line file of stub classes was added so that the pickle would *load*. Each stub's `transform()` returned its input unchanged. |
| Detection | A full code review found that `transform()` returned 8 columns where the model expects 88. Every prediction raised `FEATURE VALIDATION FAILED`. In Spark each record became `decision='ERROR'`, which neither output topic accepts, so the dashboard received nothing. The direct API returned HTTP 500. |

## Root cause

1. **The pickle stores state, not code.** `feature_pipeline.pkl` stores the fitted attributes (`freq_maps`, `card1_amt_mean`, `feature_names`, …). The `transform()` logic is resolved at load time from whatever module is named `feature_pipeline` on `sys.path`. Swapping in a stub with the same class name made the artifact load without error while doing nothing.
2. **Two copies of the module.** Training (`src/dags/`) and inference (`src/inference/`) each had their own `feature_pipeline.py`, and they drifted apart.
3. **No test exercised the serving path.** The verification scripts printed results but asserted nothing, and nothing ran them automatically.

Two subtler skews were found while fixing it:

- **Batch-dependent features.** When encoding maps were missing, the pipeline counted values *inside the incoming batch*. It also treated an absent field differently from a field that was present but empty, and treated JSON `null` (`None`) differently from `NaN`. The same transaction therefore scored 0.01854 through the REST API (a batch of one) and 0.01842 in a Spark micro-batch.
- **Wrong config in Docker.** The working directory `/app` contained the *training* `config.yaml`, which shadowed the inference config. Thresholds and Redis settings silently fell back to code defaults.

## Fix

- One shared module, `src/dags/feature_pipeline.py`, restored from commit `7624d67`. Training and both serving paths import it. The inference image copies it from `src/dags/` instead of keeping its own copy.
- One shared scorer, `src/inference/scoring.py` (`FraudScorer`). The Spark job and the REST API now run the same feature pipeline, model, velocity/ATO checks and decision engine.
- Normalisation in `transform()`: absent field ≡ `NaN` ≡ `None`, and no statistics are computed within a batch.
- `Config.load()` prefers the module-local inference config. `INFERENCE_CONFIG_PATH` overrides it.
- Tests (`tests/`):
  - `test_feature_parity.py`: the real class is loaded, the output columns equal the model's `feature_names` in order, no values are NaN, and features do not depend on what else is in the batch.
  - `test_replay_scoring.py`: single scoring and batch scoring give identical probabilities and decisions.
  - `test_model_loading.py`: input with missing features is rejected rather than scored.

## Follow-up: remove the root causes

The artifact that was in production predated the full pipeline. Its `feature_pipeline.pkl` lacked the Magic-UID, group-aggregate and encoding maps, so about half the features were served as defaults. The training code also computed those aggregates over the *whole* dataset, test period included. The follow-up therefore changed the design rather than patching the artifact:

- **The pipeline fits on the training period only.** `IEEECISFeaturePipeline.fit()` learns every map. `transform()` is the only feature code used for training, early-stop, calibration, test and serving. Training rows get out-of-fold target encoding.
- **A decision-time model for live traffic.** IEEE-CIS's anonymised V/C/D/id columns cannot be produced by an authorization request. The served model is therefore trained on simulated transactions whose features come from `decision_time_features.py`: offline via an in-memory store, online via Redis, with one feature definition. `tests/test_decision_time_features.py` asserts that both stores produce identical features.
- **Delayed labels.** Terminal-risk features only see frauds after their chargeback arrives, offline (replay) and online (`POST /feedback`).

**Remaining work:** retrain both models, then commit the generated `metrics_*.json` and a latency benchmark.

## Lessons

- Keep feature code in exactly one place and import it from both training and serving (Google, *Rules of ML* #32).
- Fail loudly. A stub that lets a pickle load is worse than an import error.
- Feature parity is a test, not a script. Run it in CI against the real artifact.
- Treat "the service is up" and "the service makes correct decisions" as separate health signals.
