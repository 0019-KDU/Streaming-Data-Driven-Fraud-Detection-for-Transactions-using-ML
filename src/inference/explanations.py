"""
Human-readable explanations for recorded decisions.

The explanation layer never decides anything. It only renders reason codes the
decision engine already produced, using fixed templates. A sentence can therefore
only mention factors that really contributed (no free-text generation that could
invent a reason a regulator or customer would later dispute).
"""

from typing import Dict, Iterable, List

# (reason-code prefix, template). {v} is the text after the prefix.
TEMPLATES = [
    ("ACCOUNT_TAKEOVER_CRITICAL", "Strong account-takeover pattern: several signals changed at once"),
    ("ACCOUNT_TAKEOVER_HIGH", "Possible account takeover: the account's usual behaviour changed"),
    ("emergency_block_terminal_", "Merchant terminal {v} is on the emergency block list"),
    ("extreme_amount_", "Amount ${v} is far above normal card limits"),
    ("night_high_amount_", "Large payment (${v}) made at night"),
    ("risky_email_", "Purchaser used a disposable email domain ({v})"),
    ("email_risky_domain_", "Purchaser used a disposable email domain ({v})"),
    ("suspicious_wire_transfer_", "Unusually large transfer-type payment (${v})"),
    ("high_1h_txn_count_", "{v} payments on this card in the last hour"),
    ("high_6h_txn_count_", "{v} payments on this card in the last 6 hours"),
    ("high_24h_txn_count_", "{v} payments on this card in the last 24 hours"),
    ("amount_spike_5x_", "Amount is {v} the card's recent average"),
    ("amount_spike_3x_", "Amount is {v} the card's recent average"),
    ("geo_anomaly_distance_", "Payment made {v} from the usual location"),
    ("geo_anomaly_new_address", "Billing address not seen before on this card"),
    ("device_anomaly_new_device", "Payment from a device not seen before on this card"),
    ("email_anomaly_new_domain", "Email domain differs from the card's usual one"),
    ("amount_anomaly_", "Amount ${v} is unusual for this card"),
    ("card_type_anomaly", "Card type differs from the usual one"),
    ("high_ml_score_", "Fraud model score is high ({v})"),
    ("medium_ml_score_", "Fraud model score is elevated ({v})"),
    ("very_high_amount_", "Very high amount (${v})"),
    ("high_amount_", "High amount (${v})"),
    ("micro_transaction_", "Very small amount (${v}), typical of card testing"),
    ("night_transaction", "Payment made at night"),
    ("model:", "Model signal: {v}"),
]
IGNORED = {"business_hours"}  # context only, not a reason for flagging

FEATURE_TEXT: Dict[str, str] = {  # decision-time model features (top contributors)
    'cust_nb_tx_1h': 'many payments in the last hour', 'cust_nb_tx_1d': 'many payments today',
    'amount_to_avg_30d': 'amount much higher than the 30-day average', 'is_new_device_30d': 'new device',
    'is_new_email_30d': 'new email domain', 'dist_from_home': 'far from home', 'dist_high': 'far from home',
    'term_risk_28d': 'merchant terminal with recent confirmed fraud', 'term_risk_7d': 'merchant terminal with recent confirmed fraud',
    'term_risk_1d': 'merchant terminal with recent confirmed fraud', 'device_nb_customers_30d': 'device shared by other customers',
    'email_risky': 'disposable email domain', 'amount': 'payment amount', 'log_amount': 'payment amount',
    'cust_amount_sum_1h': 'high spend in the last hour', 'secs_since_last_tx': 'unusual time since previous payment',
    'cust_nb_devices_7d': 'several devices used this week', 'is_night': 'night-time payment',
}


def explain_factor(code: str) -> str:
    for prefix, template in TEMPLATES:
        if code == prefix or code.startswith(prefix):
            value = code[len(prefix):]
            if prefix == "model:":
                value = FEATURE_TEXT.get(value, value)
            return template.format(v=value)
    return code.replace('_', ' ')


def explain(decision: str, factors: Iterable[str], limit: int = 4) -> List[str]:
    """Sentences for the strongest recorded factors (empty for clean approvals)."""
    codes = [f for f in factors if f and f not in IGNORED]
    if not codes:
        return [] if decision == "APPROVE" else ["Flagged by the combined risk score"]
    seen, out = set(), []
    for code in codes:
        text = explain_factor(code)
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out[:limit]
