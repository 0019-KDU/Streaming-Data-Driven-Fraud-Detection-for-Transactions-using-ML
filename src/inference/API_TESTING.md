# Scoring API: testing guide

`api_direct.py` is the **synchronous authorization path**. It returns a decision in the
request/response cycle, using the same `FraudScorer` as the Spark streaming job.
In Docker it runs as the `scoring-api` service on port **8001**.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness + whether the model is loaded |
| `POST /predict` | Score one transaction: decision, probability, reason codes, explanation |
| `POST /feedback` | Report a confirmed outcome (chargeback / analyst decision) for a past transaction |

## Authentication, latency budget, fallback

- Every call to `/predict` and `/feedback` needs `X-API-Key`. Valid keys are listed in `SCORING_API_KEYS` (if it is empty, the API is open; development only).
- If the model fails, or misses the `SCORING_TIMEOUT_MS` budget (default 300 ms), the response is a **rules-only decision** with `"degraded": true`, not an error. `FALLBACK_MODE=review` (default) sends clean payments to review; `approve` keeps checkout working.
- Retrying the same `TransactionID` returns the original decision with `"replayed": true` and is not counted again in velocity.

## Run

```bash
cd src && docker compose up -d redis postgres scoring-api
# or locally (Python 3.10, from src/):  python inference/api_direct.py
```

The default live model is the **decision-time model** (`/app/models/decision_time_model.pkl`).
Train it first with the `decision_time_fraud_training` Airflow DAG, or with
`python src/dags/train_decision_time.py --config <config>`.

## Example: normal payment

```bash
curl -s -X POST http://localhost:8001/predict -H "Content-Type: application/json" -H "X-API-Key: $FRAUD_API_KEY" -d '{
  "TransactionID": "T1", "tx_datetime": "2026-01-15T12:00:00Z",
  "customer_id": 42, "card1": 42, "terminal_id": 311,
  "TransactionAmt": 35.50, "P_emaildomain": "gmail.com", "DeviceInfo": "dev_42_0",
  "card4": "visa", "card6": "debit", "dist1": 4.2
}'
```

## Example: account-takeover pattern

Same customer, a new device shared with other victims, a disposable email, far from home:

```bash
curl -s -X POST http://localhost:8001/predict -H "Content-Type: application/json" -H "X-API-Key: $FRAUD_API_KEY" -d '{
  "TransactionID": "T2", "tx_datetime": "2026-01-15T12:05:00Z",
  "customer_id": 42, "card1": 42, "terminal_id": 9876,
  "TransactionAmt": 1850.00, "P_emaildomain": "tempmail.com", "DeviceInfo": "attacker_3",
  "card4": "visa", "card6": "debit", "dist1": 2400
}'
```

The response contains `decision` (APPROVE / REVIEW / HOLD / BLOCK), `fraud_probability`,
`risk_factors` (reason codes), `explanation` (human-readable sentences built only from those
codes), and `model_version` / `rules_version` for the audit trail.

## Example: chargeback feedback

```bash
curl -s -X POST http://localhost:8001/feedback -H "Content-Type: application/json" -H "X-API-Key: $FRAUD_API_KEY" -d '{
  "transaction_id": "T2", "terminal_id": "9876", "tx_datetime": "2026-01-15T12:05:00Z", "is_fraud": true
}'
```

Later payments at terminal 9876 now see a higher `term_risk_*` feature.

## Demo traffic and latency

```bash
python scripts/stream_synthetic.py --target predict --attack card_testing   # live attack demo
python scripts/benchmark_latency.py --n 2000 --concurrency 8                # p50/p95/p99 -> results/
```

## Emergency rule (no restart)

Edit `src/inference/rules.yaml`, set `enabled: true` on `emergency_block_terminal`, put the
terminal id in the list, and save. Within about 5 seconds, payments at that terminal return `BLOCK`
with reason `emergency_block_terminal_<id>`.

## Audit trail

Every decision is written to Postgres (`fraud` database, `decision_log` table):

```bash
docker compose exec postgres psql -U fraud -d fraud -c \
  "select decided_at, transaction_id, decision, fraud_probability, risk_factors, explanation, model_version
   from decision_log order by decided_at desc limit 5;"
```
