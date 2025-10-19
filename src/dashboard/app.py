"""
Real-time Fraud Detection Dashboard

Streamlit dashboard that:
- Subscribes to fraud_predictions and legit_predictions Kafka topics
- Displays real-time transaction table with color-coded risk levels
- Shows transaction statistics and fraud rate
- Displays success/block indicators with visual feedback
"""

import json
import logging
import os
import time
from datetime import datetime
from collections import deque
from typing import Dict, List, Any

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from confluent_kafka import Consumer, KafkaError
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path="/app/.env")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== Kafka Consumer Setup =====

@st.cache_resource
def initialize_kafka_consumers():
    """Initialize Kafka consumers for fraud and legit predictions"""
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    kafka_username = os.getenv("KAFKA_USERNAME")
    kafka_password = os.getenv("KAFKA_PASSWORD")

    consumer_config = {
        "bootstrap.servers": bootstrap_servers,
        "group.id": "dashboard-consumer-group",
        "auto.offset.reset": "latest",
        "enable.auto.commit": True,
    }

    if kafka_username and kafka_password:
        consumer_config.update({
            "security.protocol": "SASL_SSL",
            "sasl.mechanism": "PLAIN",
            "sasl.username": kafka_username,
            "sasl.password": kafka_password,
        })
    else:
        consumer_config["security.protocol"] = "PLAINTEXT"

    # Create consumers for both topics
    fraud_consumer = Consumer(consumer_config)
    fraud_consumer.subscribe(["fraud_predictions"])

    legit_config = consumer_config.copy()
    legit_config["group.id"] = "dashboard-legit-consumer-group"
    legit_consumer = Consumer(legit_config)
    legit_consumer.subscribe(["legit_predictions"])

    logger.info("Kafka consumers initialized")
    return fraud_consumer, legit_consumer


# ===== Session State Initialization =====

if 'transactions' not in st.session_state:
    st.session_state.transactions = deque(maxlen=1000)  # Keep last 1000 transactions

if 'fraud_count' not in st.session_state:
    st.session_state.fraud_count = 0

if 'legit_count' not in st.session_state:
    st.session_state.legit_count = 0

if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()


# ===== Helper Functions =====

def consume_messages(fraud_consumer, legit_consumer, max_messages=50):
    """
    Consume messages from both Kafka topics

    Args:
        fraud_consumer: Kafka consumer for fraud_predictions
        legit_consumer: Kafka consumer for legit_predictions
        max_messages: Maximum number of messages to consume per call

    Returns:
        Tuple of (fraud_messages, legit_messages)
    """
    fraud_messages = []
    legit_messages = []

    # Consume fraud predictions
    for _ in range(max_messages):
        msg = fraud_consumer.poll(timeout=0.1)
        if msg is None:
            break
        if not msg.error():
            try:
                data = json.loads(msg.value().decode('utf-8'))
                data['type'] = 'FRAUD'
                fraud_messages.append(data)
            except Exception as e:
                logger.error(f"Error parsing fraud message: {e}")

    # Consume legit predictions
    for _ in range(max_messages):
        msg = legit_consumer.poll(timeout=0.1)
        if msg is None:
            break
        if not msg.error():
            try:
                data = json.loads(msg.value().decode('utf-8'))
                data['type'] = 'LEGIT'
                legit_messages.append(data)
            except Exception as e:
                logger.error(f"Error parsing legit message: {e}")

    return fraud_messages, legit_messages


def get_risk_color(risk_level: str) -> str:
    """Get color for risk level"""
    colors = {
        'HIGH': '#ff4444',
        'MEDIUM': '#ff9800',
        'LOW': '#4caf50'
    }
    return colors.get(risk_level, '#888888')


def format_transaction_row(txn: Dict[str, Any]) -> Dict[str, Any]:
    """Format transaction for display"""
    return {
        'Transaction ID': txn.get('transaction_id', 'N/A'),
        'Type': txn.get('type', 'UNKNOWN'),
        'Decision': txn.get('decision', 'N/A'),
        'Risk Level': txn.get('risk_level', 'N/A'),
        'Probability': f"{txn.get('probability', 0.0):.3f}",
        'Amount': f"${txn.get('amount', 0.0):.2f}",
        'Timestamp': txn.get('timestamp', 'N/A'),
        'User ID': txn.get('user_id', 'N/A'),
        'Merchant': txn.get('merchant', 'N/A'),
        'Location': txn.get('location', 'N/A')
    }


# ===== Main Dashboard =====

def main():
    """Main dashboard application"""

    # Title and header
    st.title("🛡️ Real-time Fraud Detection Dashboard")
    st.markdown("---")

    # Initialize Kafka consumers
    fraud_consumer, legit_consumer = initialize_kafka_consumers()

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        refresh_rate = st.slider("Refresh Rate (seconds)", 1, 10, 2)
        max_display = st.slider("Max Transactions Displayed", 10, 200, 100)

        st.markdown("---")
        st.header("📊 Statistics")

        total_transactions = st.session_state.fraud_count + st.session_state.legit_count
        fraud_rate = (
            st.session_state.fraud_count / total_transactions * 100
            if total_transactions > 0 else 0
        )

        st.metric("Total Transactions", total_transactions)
        st.metric("Fraud Detected", st.session_state.fraud_count, delta=None)
        st.metric("Legitimate", st.session_state.legit_count, delta=None)
        st.metric("Fraud Rate", f"{fraud_rate:.2f}%", delta=None)

        st.markdown("---")
        st.caption(f"Last Update: {st.session_state.last_update.strftime('%H:%M:%S')}")

        # Manual refresh button
        if st.button("🔄 Refresh Now"):
            st.rerun()

    # Consume new messages
    fraud_messages, legit_messages = consume_messages(fraud_consumer, legit_consumer)

    # Update session state
    if fraud_messages:
        st.session_state.fraud_count += len(fraud_messages)
        st.session_state.transactions.extend(fraud_messages)

    if legit_messages:
        st.session_state.legit_count += len(legit_messages)
        st.session_state.transactions.extend(legit_messages)

    st.session_state.last_update = datetime.now()

    # Display alerts for recent fraud
    if fraud_messages:
        st.error(f"🚨 **{len(fraud_messages)} FRAUDULENT TRANSACTION(S) DETECTED!**")
        for fraud_txn in fraud_messages[:3]:  # Show first 3
            with st.expander(f"🔴 Transaction {fraud_txn.get('transaction_id', 'N/A')}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Risk Level:** {fraud_txn.get('risk_level', 'N/A')}")
                    st.write(f"**Probability:** {fraud_txn.get('probability', 0.0):.3f}")
                    st.write(f"**Amount:** ${fraud_txn.get('amount', 0.0):.2f}")
                with col2:
                    st.write(f"**Decision:** {fraud_txn.get('decision', 'N/A')}")
                    st.write(f"**User ID:** {fraud_txn.get('user_id', 'N/A')}")
                    st.write(f"**Merchant:** {fraud_txn.get('merchant', 'N/A')}")

    # Display success for recent legit transactions
    if legit_messages and len(legit_messages) >= 5:
        st.success(f"✅ **{len(legit_messages)} legitimate transaction(s) approved**")

    st.markdown("---")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🔴 Fraud (Last Minute)",
            len([t for t in st.session_state.transactions if t.get('type') == 'FRAUD'][-60:])
        )

    with col2:
        st.metric(
            "✅ Legit (Last Minute)",
            len([t for t in st.session_state.transactions if t.get('type') == 'LEGIT'][-60:])
        )

    with col3:
        recent_fraud_probs = [
            t.get('probability', 0.0)
            for t in st.session_state.transactions
            if t.get('type') == 'FRAUD'
        ][-10:]
        avg_fraud_prob = sum(recent_fraud_probs) / len(recent_fraud_probs) if recent_fraud_probs else 0
        st.metric("Avg Fraud Probability", f"{avg_fraud_prob:.3f}")

    with col4:
        recent_amounts = [
            t.get('amount', 0.0)
            for t in st.session_state.transactions
        ][-10:]
        avg_amount = sum(recent_amounts) / len(recent_amounts) if recent_amounts else 0
        st.metric("Avg Transaction Amount", f"${avg_amount:.2f}")

    st.markdown("---")

    # Transactions table
    st.subheader("📋 Recent Transactions")

    if st.session_state.transactions:
        # Convert to DataFrame
        recent_txns = list(st.session_state.transactions)[-max_display:]
        recent_txns.reverse()  # Show newest first

        df = pd.DataFrame([format_transaction_row(txn) for txn in recent_txns])

        # Style the dataframe
        def highlight_fraud(row):
            if row['Type'] == 'FRAUD':
                return ['background-color: #ffe6e6'] * len(row)
            elif row['Type'] == 'LEGIT':
                return ['background-color: #e6ffe6'] * len(row)
            return [''] * len(row)

        styled_df = df.style.apply(highlight_fraud, axis=1)

        st.dataframe(styled_df, use_container_width=True, height=400)

    else:
        st.info("⏳ Waiting for transactions...")

    st.markdown("---")

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Transaction Volume (Last 100)")

        if st.session_state.transactions:
            recent_100 = list(st.session_state.transactions)[-100:]
            fraud_vol = [t for t in recent_100 if t.get('type') == 'FRAUD']
            legit_vol = [t for t in recent_100 if t.get('type') == 'LEGIT']

            fig = go.Figure(data=[
                go.Bar(name='Fraud', x=['Transactions'], y=[len(fraud_vol)], marker_color='#ff4444'),
                go.Bar(name='Legit', x=['Transactions'], y=[len(legit_vol)], marker_color='#4caf50')
            ])

            fig.update_layout(
                barmode='group',
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data yet")

    with col2:
        st.subheader("📈 Risk Level Distribution")

        if st.session_state.transactions:
            recent_100 = list(st.session_state.transactions)[-100:]
            risk_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}

            for txn in recent_100:
                risk_level = txn.get('risk_level', 'LOW')
                risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1

            fig = go.Figure(data=[
                go.Pie(
                    labels=list(risk_counts.keys()),
                    values=list(risk_counts.values()),
                    marker=dict(colors=['#ff4444', '#ff9800', '#4caf50']),
                    hole=0.3
                )
            ])

            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data yet")

    # Auto-refresh
    time.sleep(refresh_rate)
    st.rerun()


if __name__ == "__main__":
    main()
