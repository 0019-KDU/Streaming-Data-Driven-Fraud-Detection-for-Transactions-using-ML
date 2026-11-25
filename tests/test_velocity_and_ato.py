"""Redis-backed velocity and ATO services, run against an in-memory fake Redis."""

from inference.ato_service import ATOService
from inference.config import Config
from inference.velocity_service import VelocityService


def test_repeated_equal_amounts_are_all_counted(fake_redis):
    """Regression: amounts were sorted-set members, so five $50 payments counted as one."""
    service = VelocityService(fake_redis, Config())
    t0 = 1_700_000_000.0
    for i in range(6):
        result = service.check_velocity("2616", "uid", 50.0, t0 + i)

    assert result.txn_count_1h == 5          # counted before recording the 6th
    assert result.amt_sum_1h == 250.0
    assert result.amt_mean_24h == 50.0
    assert result.velocity_risk >= 0.7      # 5 txns/hour hits the high-velocity band


def test_old_events_fall_out_of_the_1h_window(fake_redis):
    service = VelocityService(fake_redis, Config())
    t0 = 1_700_000_000.0
    service.check_velocity("1", "u", 10.0, t0)
    result = service.check_velocity("1", "u", 10.0, t0 + 2 * 3600)
    assert result.txn_count_1h == 0
    assert result.txn_count_24h == 1


def test_ato_reads_raw_ieee_field_names(fake_redis):
    """Regression: the service looked for email_domain/amount/card_type and ignored the payload."""
    service = ATOService(fake_redis, Config())
    raw = {"P_emaildomain": "tempmail.com", "TransactionAmt": 2500.0, "card6": "credit", "addr1": 315.0}

    result = service.check_ato("2616", raw)

    assert result.email_anomaly > 0.5       # risky disposable domain
    assert result.amount_anomaly > 0.5      # above the 2000 high-amount threshold
    assert "email_risky_domain_tempmail.com" in result.factors


def test_ato_flags_takeover_pattern_against_baseline(fake_redis):
    service = ATOService(fake_redis, Config())
    normal = {"P_emaildomain": "gmail.com", "TransactionAmt": 40.0, "card6": "debit", "addr1": 315.0, "dist1": 5.0}
    for _ in range(5):
        service.check_ato("8026", normal)

    takeover = {"P_emaildomain": "tempmail.com", "TransactionAmt": 2500.0, "card6": "credit",
                "addr1": 589.0, "dist1": 1500.0}
    result = service.check_ato("8026", takeover)

    assert result.ato_detected
    assert result.ato_risk >= 0.6


def test_normalize_drops_nan_and_keeps_internal_names():
    out = ATOService.normalize_transaction({"P_emaildomain": float("nan"), "email_domain": "x.com", "TransactionAmt": "12.5"})
    assert out["email_domain"] == "x.com"
    assert out["amount"] == 12.5
    assert "P_emaildomain" not in out
