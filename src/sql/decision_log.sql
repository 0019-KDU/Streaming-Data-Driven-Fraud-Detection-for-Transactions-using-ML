-- Decision audit log: one row per scored transaction (see src/inference/audit_log.py).
-- Answers "why was transaction X declined?" months or years later: inputs, features,
-- scores, thresholds, reason codes, explanation and the exact model + rule versions.
CREATE TABLE IF NOT EXISTS decision_log (
    id                    BIGSERIAL PRIMARY KEY,
    decided_at            TIMESTAMPTZ NOT NULL,
    transaction_id        TEXT        NOT NULL,
    serving_path          TEXT        NOT NULL,          -- sync_api | spark_stream
    model_kind            TEXT,
    model_version         TEXT,
    rules_version         TEXT,
    decision              TEXT        NOT NULL,          -- APPROVE | REVIEW | HOLD | BLOCK
    risk_level            TEXT,
    fraud_probability     DOUBLE PRECISION,
    adjusted_probability  DOUBLE PRECISION,
    hybrid_threshold      DOUBLE PRECISION,
    shadow_model_version  TEXT,
    shadow_probability    DOUBLE PRECISION,
    risk_factors          TEXT[],
    explanation           TEXT,
    features              JSONB,
    request               JSONB
);
CREATE INDEX IF NOT EXISTS decision_log_txn_idx  ON decision_log (transaction_id);
CREATE INDEX IF NOT EXISTS decision_log_time_idx ON decision_log (decided_at);
CREATE INDEX IF NOT EXISTS decision_log_decision_idx ON decision_log (decision, decided_at);
