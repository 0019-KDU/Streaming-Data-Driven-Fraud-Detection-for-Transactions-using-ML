"""
Latency benchmark for the synchronous scoring path (POST /predict).

Reports client-observed p50/p95/p99/max latency and throughput, plus the
server-side processing_time_ms, so network/serialisation overhead is visible.
Writes results/latency_<timestamp>.json. Quote these numbers, with the hardware,
instead of unmeasured claims.

  python scripts/benchmark_latency.py --n 2000 --concurrency 8
"""

import argparse
import json
import os
import platform
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests


def payload(i: int) -> dict:
    return {
        'TransactionID': f"BENCH_{i}", 'tx_datetime': datetime.now(timezone.utc).isoformat(),
        'customer_id': random.randint(0, 999), 'card1': random.randint(0, 999),
        'terminal_id': random.randint(0, 1999), 'TransactionAmt': round(random.uniform(1, 300), 2),
        'P_emaildomain': random.choice(['gmail.com', 'yahoo.com', 'tempmail.com']),
        'DeviceInfo': f"dev_{random.randint(0, 999)}", 'card4': 'visa', 'card6': 'debit',
        'dist1': round(random.uniform(0, 60), 1),
    }


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q / 100 * (len(values) - 1))))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', default=os.getenv('FRAUD_SCORING_BASE', 'http://localhost:8001') + '/predict')
    p.add_argument('--n', type=int, default=1000)
    p.add_argument('--concurrency', type=int, default=4)
    p.add_argument('--warmup', type=int, default=50)
    args = p.parse_args()

    session = requests.Session()
    session.headers['X-API-Key'] = os.getenv('FRAUD_API_KEY', '')
    for i in range(args.warmup):
        session.post(args.url, json=payload(-i), timeout=10)

    def one(i):
        t0 = time.perf_counter()
        r = session.post(args.url, json=payload(i), timeout=10)
        elapsed = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        body = r.json()
        return elapsed, body.get('processing_time_ms', float('nan')), bool(body.get('degraded'))

    start = time.perf_counter()
    with ThreadPoolExecutor(args.concurrency) as pool:
        results = list(pool.map(one, range(args.n)))
    wall = time.perf_counter() - start

    client = [r[0] for r in results]
    server = [r[1] for r in results]
    degraded = sum(r[2] for r in results)
    report = {
        'url': args.url, 'n': args.n, 'concurrency': args.concurrency,
        'throughput_tps': args.n / wall,
        'degraded_responses': degraded,  # rules-only fallbacks (model timeout/error) during the run
        'client_ms': {f'p{q}': percentile(client, q) for q in (50, 95, 99)} | {'max': max(client),
                                                                             'mean': statistics.mean(client)},
        'server_ms': {f'p{q}': percentile(server, q) for q in (50, 95, 99)},
        'host': {'platform': platform.platform(), 'processor': platform.processor(), 'cpus': os.cpu_count()},
        'measured_at': datetime.now(timezone.utc).isoformat(),
    }
    os.makedirs('results', exist_ok=True)
    path = os.path.join('results', f"latency_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json")
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"saved {path}")


if __name__ == '__main__':
    main()
