"""
Live demo traffic: replay simulated transactions "now", with optional attacks.

Sends payloads with only the fields a real authorization request carries (never
the label) to either the async path (producer API -> Kafka -> Spark) or the sync
scoring API (/predict). Confirmed frauds are reported back later to /feedback,
like chargebacks arriving after the fact.

Examples (from the repo root):
  python scripts/stream_synthetic.py --target predict --rate 20 --n 500
  python scripts/stream_synthetic.py --target kafka --attack card_testing
  python scripts/stream_synthetic.py --target predict --attack ato --n 50
"""

import argparse
import os
import random
import sys
import threading
import time
from datetime import datetime, timezone

import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'dags'))
from simulator import DISPOSABLE, SimConfig, simulate  # noqa: E402

PAYLOAD_FIELDS = ['customer_id', 'card1', 'terminal_id', 'TransactionAmt', 'P_emaildomain',
                  'DeviceInfo', 'card4', 'card6', 'dist1']


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


HEADERS = {'X-API-Key': os.getenv('FRAUD_API_KEY', '')}


def send(args, payload: dict) -> dict:
    if args.target == 'predict':
        r = requests.post(f"{args.scoring_base}/predict", json=payload, headers=HEADERS, timeout=5)
    else:
        r = requests.post(f"{args.api_base}/api/v1/transactions/submit", json=payload, headers=HEADERS, timeout=5)
    r.raise_for_status()
    return r.json()


def report_chargeback(args, payload: dict) -> None:
    """Simulated chargeback: arrives `--chargeback-delay` seconds after the payment."""
    def later():
        time.sleep(args.chargeback_delay)
        try:
            requests.post(f"{args.scoring_base}/feedback", timeout=5, headers=HEADERS, json={
                'transaction_id': payload['TransactionID'], 'terminal_id': payload['terminal_id'],
                'tx_datetime': payload['tx_datetime'], 'is_fraud': True})
        except requests.RequestException as exc:
            print(f"  feedback failed: {exc}")
    threading.Thread(target=later, daemon=True).start()


def attack_payloads(kind: str, base: dict) -> list:
    """Attack built on a real customer's profile."""
    out = []
    if kind == 'card_testing':
        device = f"attacker_{random.randint(0, 24)}"
        for _ in range(random.randint(5, 10)):
            out.append(dict(base, TransactionAmt=round(random.uniform(0.5, 2.0), 2), DeviceInfo=device))
        out.append(dict(base, TransactionAmt=round(random.uniform(500, 1500), 2), DeviceInfo=device))
    elif kind == 'ato':
        device, email = f"attacker_{random.randint(0, 24)}", random.choice(DISPOSABLE)
        for _ in range(random.randint(3, 6)):
            out.append(dict(base, TransactionAmt=round(random.uniform(300, 2500), 2), DeviceInfo=device,
                            P_emaildomain=email, dist1=round(random.uniform(1000, 4000), 1),
                            terminal_id=random.randint(0, 9999)))
    elif kind == 'terminal':
        for _ in range(10):
            out.append(dict(base, customer_id=random.randint(0, 499), terminal_id=1234))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--target', choices=['kafka', 'predict'], default='predict')
    p.add_argument('--api-base', default=os.getenv('FRAUD_API_BASE', 'http://localhost:8000'))
    p.add_argument('--scoring-base', default=os.getenv('FRAUD_SCORING_BASE', 'http://localhost:8001'))
    p.add_argument('--rate', type=float, default=10.0, help='transactions per second')
    p.add_argument('--n', type=int, default=300)
    p.add_argument('--attack', choices=['none', 'card_testing', 'ato', 'terminal'], default='none')
    p.add_argument('--chargeback-delay', type=float, default=30.0, help='seconds until a fraud is reported')
    p.add_argument('--seed', type=int, default=7)
    args = p.parse_args()
    random.seed(args.seed)

    print("Simulating background traffic...")
    data = simulate(SimConfig(n_customers=500, n_terminals=1000, n_days=14, seed=args.seed))
    rows = data.sample(n=min(args.n, len(data)), random_state=args.seed).to_dict(orient='records')

    queue = []
    for i, row in enumerate(rows):
        payload = {k: (int(row[k]) if k in ('customer_id', 'card1', 'terminal_id') else row[k]) for k in PAYLOAD_FIELDS}
        queue.append((payload, bool(row['is_fraud'])))
        if args.attack != 'none' and i == len(rows) // 2:
            queue.extend((a, True) for a in attack_payloads(args.attack, payload))
            print(f"Injected {args.attack} attack at position {i}")

    sent = flagged = 0
    for payload, is_fraud in queue:
        payload = dict(payload, TransactionID=f"SIM_{int(time.time() * 1e6)}", tx_datetime=now_iso())
        try:
            result = send(args, payload)
        except requests.RequestException as exc:
            print(f"send failed: {exc}")
            continue
        sent += 1
        if args.target == 'predict' and result.get('decision') != 'APPROVE':
            flagged += 1
            print(f"{result['decision']:<7} ${payload['TransactionAmt']:>8.2f}  p={result['fraud_probability']:.3f}  "
                  f"{'; '.join(result.get('explanation', []))}")
        if is_fraud:
            report_chargeback(args, payload)
        time.sleep(1.0 / args.rate)

    print(f"Sent {sent} transactions" + (f", {flagged} flagged" if args.target == 'predict' else ""))
    if any(f for _, f in queue):
        print(f"Waiting {args.chargeback_delay:.0f}s for simulated chargebacks to be reported...")
        time.sleep(args.chargeback_delay + 2)


if __name__ == '__main__':
    main()
