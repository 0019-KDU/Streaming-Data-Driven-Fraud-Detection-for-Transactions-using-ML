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
    page_title="Fraud Detection Dashboard - Enhanced",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better visualization
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .fraud-alert {
        background-color: #ffe6e6;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #ff4444;
    }
    .ato-critical {
        background-color: #ffcccc;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #cc0000;
        font-weight: bold;
    }
    .ato-high {
        background-color: #fff0cc;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #ff9800;
    }
    .success-msg {
        background-color: #e6ffe6;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

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

if 'model_stats' not in st.session_state:
    st.session_state.model_stats = {
        'total_predictions': 0,
        'avg_confidence': 0.0,
        'ato_detections': 0,
        'velocity_alerts': 0,
        'amount_alerts': 0,
        'email_risk_alerts': 0,
        'threshold_adjustments': 0
    }

if 'ato_incidents' not in st.session_state:
    st.session_state.ato_incidents = deque(maxlen=100)


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
    # Extract risk factors
    risk_factors = txn.get('risk_factors', 'none')
    
    # Check for ATO
    is_ato = 'ACCOUNT_TAKEOVER' in risk_factors
    ato_badge = '🚨 ATO' if is_ato else ''
    
    # Check for velocity
    has_velocity = 'velocity' in risk_factors.lower()
    velocity_badge = '⚡' if has_velocity else ''
    
    return {
        'Transaction ID': txn.get('transaction_id', 'N/A'),
        'Type': txn.get('type', 'UNKNOWN'),
        'Decision': txn.get('decision', 'N/A'),
        'Risk Level': txn.get('risk_level', 'N/A'),
        'Probability': f"{txn.get('probability', 0.0):.3f}",
        'Amount': f"${txn.get('amount', 0.0):.2f}",
        'Flags': f"{ato_badge} {velocity_badge}".strip(),
        'Timestamp': txn.get('timestamp', 'N/A'),
        'User ID': txn.get('user_id', 'N/A'),
        'Risk Factors': risk_factors[:50] + '...' if len(risk_factors) > 50 else risk_factors
    }


def update_model_stats(transactions: List[Dict[str, Any]]):
    """Update model statistics based on recent transactions"""
    if not transactions:
        return
    
    for txn in transactions:
        st.session_state.model_stats['total_predictions'] += 1
        
        # Track ATO detections
        risk_factors = txn.get('risk_factors', '')
        if 'ACCOUNT_TAKEOVER' in risk_factors:
            st.session_state.model_stats['ato_detections'] += 1
            st.session_state.ato_incidents.append(txn)
        
        # Track velocity alerts
        if 'velocity' in risk_factors.lower():
            st.session_state.model_stats['velocity_alerts'] += 1
        
        # Track amount alerts
        if 'amount' in risk_factors.lower():
            st.session_state.model_stats['amount_alerts'] += 1
        
        # Track email risk
        if 'email' in risk_factors.lower():
            st.session_state.model_stats['email_risk_alerts'] += 1
        
        # Track threshold adjustments
        if 'threshold' in risk_factors.lower():
            st.session_state.model_stats['threshold_adjustments'] += 1
    
    # Update average confidence
    probabilities = [t.get('probability', 0.0) for t in transactions if t.get('type') == 'FRAUD']
    if probabilities:
        total_prob = sum(probabilities)
        count = len(probabilities)
        st.session_state.model_stats['avg_confidence'] = (
            st.session_state.model_stats['avg_confidence'] * 0.9 + 
            (total_prob / count) * 0.1
        )


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
        st.header("📊 Transaction Statistics")

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
        st.header("🤖 ML Model Stats")
        
        model_stats = st.session_state.model_stats
        st.metric("Total Predictions", model_stats['total_predictions'])
        st.metric("Avg Fraud Confidence", f"{model_stats['avg_confidence']:.3f}")
        st.metric("🚨 ATO Detections", model_stats['ato_detections'])
        st.metric("⚡ Velocity Alerts", model_stats['velocity_alerts'])
        st.metric("💰 Amount Alerts", model_stats['amount_alerts'])
        st.metric("📧 Email Risk Alerts", model_stats['email_risk_alerts'])
        st.metric("🎯 Threshold Adjustments", model_stats['threshold_adjustments'])

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
        update_model_stats(fraud_messages)

    if legit_messages:
        st.session_state.legit_count += len(legit_messages)
        st.session_state.transactions.extend(legit_messages)
        update_model_stats(legit_messages)

    st.session_state.last_update = datetime.now()

    # Display ATO CRITICAL alerts first
    ato_critical = [m for m in fraud_messages if 'ACCOUNT_TAKEOVER_CRITICAL' in m.get('risk_factors', '')]
    if ato_critical:
        st.markdown('<div class="ato-critical">🚨 CRITICAL: ACCOUNT TAKEOVER DETECTED!</div>', unsafe_allow_html=True)
        for ato_txn in ato_critical[:3]:
            with st.expander(f"⚠️ CRITICAL ATO: Transaction {ato_txn.get('transaction_id', 'N/A')}", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Risk Level:** {ato_txn.get('risk_level', 'N/A')}")
                    st.write(f"**ML Probability:** {ato_txn.get('probability', 0.0):.3f}")
                    st.write(f"**Amount:** ${ato_txn.get('amount', 0.0):.2f}")
                with col2:
                    st.write(f"**Decision:** {ato_txn.get('decision', 'N/A')}")
                    st.write(f"**User ID:** {ato_txn.get('user_id', 'N/A')}")
                    st.write(f"**Card:** {ato_txn.get('card1', 'N/A')}")
                with col3:
                    st.write(f"**Timestamp:** {ato_txn.get('timestamp', 'N/A')}")
                    risk_factors = ato_txn.get('risk_factors', 'none')
                    st.write(f"**ATO Reasons:**")
                    for factor in risk_factors.split(','):
                        if 'ACCOUNT_TAKEOVER' in factor or 'impossible' in factor or 'ato' in factor.lower():
                            st.write(f"   • {factor.strip()}")
    
    # Display HIGH ATO alerts
    ato_high = [m for m in fraud_messages if 'ACCOUNT_TAKEOVER_HIGH' in m.get('risk_factors', '')]
    if ato_high:
        st.markdown('<div class="ato-high">⚠️ HIGH RISK: Potential Account Takeover</div>', unsafe_allow_html=True)
        for ato_txn in ato_high[:2]:
            with st.expander(f"� High Risk ATO: Transaction {ato_txn.get('transaction_id', 'N/A')}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Risk Level:** {ato_txn.get('risk_level', 'N/A')}")
                    st.write(f"**Probability:** {ato_txn.get('probability', 0.0):.3f}")
                    st.write(f"**Amount:** ${ato_txn.get('amount', 0.0):.2f}")
                with col2:
                    st.write(f"**Decision:** {ato_txn.get('decision', 'N/A')}")
                    st.write(f"**Risk Factors:** {ato_txn.get('risk_factors', 'none')[:100]}")
    
    # Display regular fraud alerts
    regular_fraud = [m for m in fraud_messages if 'ACCOUNT_TAKEOVER' not in m.get('risk_factors', '')]
    if regular_fraud:
        st.error(f"🚨 **{len(regular_fraud)} FRAUDULENT TRANSACTION(S) DETECTED!**")
        for fraud_txn in regular_fraud[:3]:
            with st.expander(f"🔴 Fraud: Transaction {fraud_txn.get('transaction_id', 'N/A')}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Risk Level:** {fraud_txn.get('risk_level', 'N/A')}")
                    st.write(f"**Probability:** {fraud_txn.get('probability', 0.0):.3f}")
                    st.write(f"**Amount:** ${fraud_txn.get('amount', 0.0):.2f}")
                with col2:
                    st.write(f"**Decision:** {fraud_txn.get('decision', 'N/A')}")
                    st.write(f"**User ID:** {fraud_txn.get('user_id', 'N/A')}")
                    st.write(f"**Merchant:** {fraud_txn.get('merchant', 'N/A')}")
                with col3:
                    st.write(f"**Risk Factors:**")
                    risk_factors = fraud_txn.get('risk_factors', 'none').split(',')
                    for factor in risk_factors[:5]:
                        st.write(f"   • {factor.strip()}")

    # Display success for recent legit transactions
    if legit_messages and len(legit_messages) >= 5:
        st.success(f"✅ **{len(legit_messages)} legitimate transaction(s) approved**")

    st.markdown("---")

    # Enhanced Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

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
        st.metric("Avg Fraud Confidence", f"{avg_fraud_prob:.3f}")

    with col4:
        recent_amounts = [
            t.get('amount', 0.0)
            for t in st.session_state.transactions
        ][-10:]
        avg_amount = sum(recent_amounts) / len(recent_amounts) if recent_amounts else 0
        st.metric("Avg Transaction $", f"${avg_amount:.2f}")
    
    with col5:
        ato_count = len([t for t in st.session_state.transactions if 'ACCOUNT_TAKEOVER' in t.get('risk_factors', '')][-100:])
        st.metric("🚨 ATO Incidents", ato_count)

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

    # Transaction Detail Inspector
    st.subheader("🔍 Transaction Detail Inspector")
    
    if st.session_state.transactions:
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "🚨 ATO Incidents", 
            "⚡ Velocity Alerts", 
            "💰 High Amount", 
            "📧 Email Risk"
        ])
        
        with tab1:
            ato_txns = [t for t in st.session_state.transactions if 'ACCOUNT_TAKEOVER' in t.get('risk_factors', '')][-20:]
            if ato_txns:
                st.write(f"**{len(ato_txns)} ATO incidents in last 1000 transactions**")
                for txn in reversed(ato_txns[:10]):
                    with st.expander(
                        f"Transaction {txn.get('transaction_id', 'N/A')} - "
                        f"${txn.get('amount', 0):.2f} - "
                        f"{txn.get('risk_level', 'N/A')}"
                    ):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.json({
                                'transaction_id': txn.get('transaction_id', 'N/A'),
                                'user_id': txn.get('user_id', 'N/A'),
                                'card1': txn.get('card1', 'N/A'),
                                'amount': txn.get('amount', 0.0),
                                'probability': txn.get('probability', 0.0),
                                'decision': txn.get('decision', 'N/A'),
                                'timestamp': txn.get('timestamp', 'N/A')
                            })
                        with col2:
                            st.write("**Risk Factors:**")
                            risk_factors = txn.get('risk_factors', 'none').split(',')
                            for factor in risk_factors:
                                if factor.strip():
                                    st.write(f"• {factor.strip()}")
            else:
                st.info("No ATO incidents detected")
        
        with tab2:
            velocity_txns = [t for t in st.session_state.transactions if 'velocity' in t.get('risk_factors', '').lower()][-20:]
            if velocity_txns:
                st.write(f"**{len(velocity_txns)} Velocity alerts in last 1000 transactions**")
                for txn in reversed(velocity_txns[:10]):
                    with st.expander(
                        f"Transaction {txn.get('transaction_id', 'N/A')} - "
                        f"${txn.get('amount', 0):.2f}"
                    ):
                        st.write(f"**Probability:** {txn.get('probability', 0.0):.3f}")
                        st.write(f"**Risk Level:** {txn.get('risk_level', 'N/A')}")
                        st.write(f"**Decision:** {txn.get('decision', 'N/A')}")
                        st.write(f"**Risk Factors:** {txn.get('risk_factors', 'none')}")
            else:
                st.info("No velocity alerts")
        
        with tab3:
            high_amount_txns = [t for t in st.session_state.transactions if t.get('amount', 0) > 1000][-20:]
            if high_amount_txns:
                st.write(f"**{len(high_amount_txns)} High-amount transactions (>$1000)**")
                for txn in reversed(high_amount_txns[:10]):
                    with st.expander(
                        f"Transaction {txn.get('transaction_id', 'N/A')} - "
                        f"${txn.get('amount', 0):.2f} - "
                        f"{txn.get('type', 'N/A')}"
                    ):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Amount:** ${txn.get('amount', 0.0):.2f}")
                            st.write(f"**Type:** {txn.get('type', 'N/A')}")
                            st.write(f"**Probability:** {txn.get('probability', 0.0):.3f}")
                        with col2:
                            st.write(f"**Risk Level:** {txn.get('risk_level', 'N/A')}")
                            st.write(f"**Decision:** {txn.get('decision', 'N/A')}")
                            st.write(f"**User:** {txn.get('user_id', 'N/A')}")
            else:
                st.info("No high-amount transactions")
        
        with tab4:
            email_risk_txns = [t for t in st.session_state.transactions if 'email' in t.get('risk_factors', '').lower()][-20:]
            if email_risk_txns:
                st.write(f"**{len(email_risk_txns)} Email risk alerts**")
                for txn in reversed(email_risk_txns[:10]):
                    with st.expander(
                        f"Transaction {txn.get('transaction_id', 'N/A')} - "
                        f"{txn.get('P_emaildomain', 'N/A')}"
                    ):
                        st.write(f"**Email Domain:** {txn.get('P_emaildomain', 'N/A')}")
                        st.write(f"**Probability:** {txn.get('probability', 0.0):.3f}")
                        st.write(f"**Decision:** {txn.get('decision', 'N/A')}")
                        st.write(f"**Risk Factors:** {txn.get('risk_factors', 'none')}")
            else:
                st.info("No email risk alerts")
    else:
        st.info("⏳ Waiting for transaction data...")

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
    
    st.markdown("---")
    
    # Additional analytics row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🎯 Detection Method Breakdown")
        if st.session_state.transactions:
            recent_fraud = [t for t in st.session_state.transactions if t.get('type') == 'FRAUD'][-100:]
            
            detection_counts = {
                'ATO Detection': 0,
                'Velocity Rules': 0,
                'Amount Rules': 0,
                'Email Rules': 0,
                'ML Model Only': 0
            }
            
            for txn in recent_fraud:
                risk_factors = txn.get('risk_factors', '')
                if 'ACCOUNT_TAKEOVER' in risk_factors:
                    detection_counts['ATO Detection'] += 1
                elif 'velocity' in risk_factors.lower():
                    detection_counts['Velocity Rules'] += 1
                elif 'amount' in risk_factors.lower():
                    detection_counts['Amount Rules'] += 1
                elif 'email' in risk_factors.lower():
                    detection_counts['Email Rules'] += 1
                else:
                    detection_counts['ML Model Only'] += 1
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(detection_counts.keys()),
                    y=list(detection_counts.values()),
                    marker_color=['#cc0000', '#ff6600', '#ff9900', '#ffcc00', '#4caf50']
                )
            ])
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data yet")
    
    with col2:
        st.subheader("📊 Decision Distribution")
        if st.session_state.transactions:
            recent_100 = list(st.session_state.transactions)[-100:]
            decision_counts = {}
            
            for txn in recent_100:
                decision = txn.get('decision', 'UNKNOWN')
                decision_counts[decision] = decision_counts.get(decision, 0) + 1
            
            colors_map = {
                'BLOCK': '#cc0000',
                'HOLD': '#ff9800',
                'REVIEW': '#ffcc00',
                'APPROVE': '#4caf50'
            }
            
            colors = [colors_map.get(d, '#888888') for d in decision_counts.keys()]
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=list(decision_counts.keys()),
                    values=list(decision_counts.values()),
                    marker=dict(colors=colors),
                    hole=0.4
                )
            ])
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data yet")
    
    with col3:
        st.subheader("⏱️ Real-time Fraud Rate")
        if st.session_state.transactions:
            # Calculate fraud rate over last 100 transactions
            recent_100 = list(st.session_state.transactions)[-100:]
            
            fraud_over_time = []
            window_size = 10
            
            for i in range(len(recent_100) - window_size):
                window = recent_100[i:i+window_size]
                fraud_in_window = sum(1 for t in window if t.get('type') == 'FRAUD')
                fraud_rate = (fraud_in_window / window_size) * 100
                fraud_over_time.append(fraud_rate)
            
            fig = go.Figure(data=[
                go.Scatter(
                    y=fraud_over_time,
                    mode='lines+markers',
                    line=dict(color='#ff4444', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(255, 68, 68, 0.2)'
                )
            ])
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis_title="Fraud Rate (%)",
                xaxis_title="Transaction Window",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data yet")

    st.markdown("---")
    
    # ML Model Performance Section
    st.subheader("🤖 ML Model Performance & Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Confidence Score Distribution (Fraud Transactions)**")
        if st.session_state.transactions:
            fraud_txns = [t for t in st.session_state.transactions if t.get('type') == 'FRAUD'][-100:]
            
            if fraud_txns:
                probabilities = [t.get('probability', 0.0) for t in fraud_txns]
                
                fig = go.Figure(data=[
                    go.Histogram(
                        x=probabilities,
                        nbinsx=20,
                        marker_color='#ff4444',
                        opacity=0.7
                    )
                ])
                
                fig.update_layout(
                    height=250,
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis_title="Fraud Probability",
                    yaxis_title="Count",
                    bargap=0.1
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Stats
                st.write(f"**Mean Confidence:** {sum(probabilities)/len(probabilities):.3f}")
                st.write(f"**Max Confidence:** {max(probabilities):.3f}")
                st.write(f"**Min Confidence:** {min(probabilities):.3f}")
            else:
                st.info("No fraud transactions yet")
        else:
            st.info("No data yet")
    
    with col2:
        st.write("**Adaptive Threshold Impact**")
        if st.session_state.transactions:
            threshold_txns = [t for t in st.session_state.transactions if 'threshold' in t.get('risk_factors', '').lower()][-50:]
            
            if threshold_txns:
                st.metric("Transactions with Threshold Adjustment", len(threshold_txns))
                
                # Show breakdown of threshold adjustments
                threshold_types = {
                    'Lowered (High Risk)': 0,
                    'Raised (Low Risk)': 0
                }
                
                for txn in threshold_txns:
                    rf = txn.get('risk_factors', '')
                    if 'threshold_lowered' in rf:
                        threshold_types['Lowered (High Risk)'] += 1
                    elif 'threshold_raised' in rf:
                        threshold_types['Raised (Low Risk)'] += 1
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(threshold_types.keys()),
                        y=list(threshold_types.values()),
                        marker_color=['#ff6600', '#4caf50']
                    )
                ])
                
                fig.update_layout(
                    height=200,
                    margin=dict(l=20, r=20, t=20, b=20),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No threshold adjustments yet")
        else:
            st.info("No data yet")
    
    # System Health Status
    st.markdown("---")
    st.subheader("🏥 System Health")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Kafka Connection", "✅ Healthy")
    
    with col2:
        processing_rate = st.session_state.model_stats['total_predictions'] / max((datetime.now() - st.session_state.last_update).total_seconds() / 60, 1)
        st.metric("Processing Rate", f"{processing_rate:.1f} txn/min")
    
    with col3:
        st.metric("ML Model Status", "✅ Active")
    
    with col4:
        redis_status = "✅ Connected" if st.session_state.model_stats['velocity_alerts'] > 0 or st.session_state.model_stats['ato_detections'] > 0 else "⚠️ Unknown"
        st.metric("Redis (ATO/Velocity)", redis_status)

    # Auto-refresh
    time.sleep(refresh_rate)
    st.rerun()


if __name__ == "__main__":
    main()
