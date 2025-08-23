import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import Header from '../../components/common/Header';
import api from '../../services/api';
import './TransactionDetail.css';

const TransactionDetail = () => {
  const { transactionId } = useParams();
  const navigate = useNavigate();
  const [transaction, setTransaction] = useState(null);
  const [customer, setCustomer] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchTransactionData();
  }, [transactionId]);

  const fetchTransactionData = async () => {
    try {
      const transactionRes = await api.get(`/transactions/${transactionId}`);
      const transactionData = transactionRes.data;
      setTransaction(transactionData);

      // Fetch customer data
      if (transactionData.user_id) {
        const customerRes = await api.get(`/customers/${transactionData.user_id}`);
        setCustomer(customerRes.data);
      }
    } catch (error) {
      console.error('Error fetching transaction data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="loading">Loading transaction details...</div>;
  }

  if (!transaction) {
    return <div className="error">Transaction not found</div>;
  }

  return (
    <div className="transaction-detail-page">
      <Header />
      <div className="transaction-detail-content">
        {/* Header */}
        <div className="transaction-detail-header">
          <button onClick={() => navigate('/transactions')} className="back-btn">
            ← Back to Transactions
          </button>
          <div className="transaction-title">
            <h1>Transaction Details</h1>
            <div className={`status-badge large ${transaction.is_fraud ? 'fraud' : 'normal'}`}>
              {transaction.is_fraud ? '⚠️ FRAUD ALERT' : '✅ NORMAL TRANSACTION'}
            </div>
          </div>
        </div>

        {/* Transaction Details Grid */}
        <div className="details-grid">
          {/* Transaction Information */}
          <div className="detail-card">
            <h3>Transaction Information</h3>
            <div className="detail-item">
              <span className="label">Transaction ID:</span>
              <span className="value transaction-id">{transaction.transaction_id}</span>
            </div>
            <div className="detail-item">
              <span className="label">Amount:</span>
              <span className="value amount">${transaction.amount.toLocaleString()}</span>
            </div>
            <div className="detail-item">
              <span className="label">Currency:</span>
              <span className="value">{transaction.currency}</span>
            </div>
            <div className="detail-item">
              <span className="label">Merchant:</span>
              <span className="value">{transaction.merchant}</span>
            </div>
            <div className="detail-item">
              <span className="label">Location:</span>
              <span className="value">{transaction.location}</span>
            </div>
          </div>

          {/* Timing Information */}
          <div className="detail-card">
            <h3>Timing Information</h3>
            <div className="detail-item">
              <span className="label">Transaction Time:</span>
              <span className="value">
                {new Date(transaction.timestamp).toLocaleString()}
              </span>
            </div>
            <div className="detail-item">
              <span className="label">Created At:</span>
              <span className="value">
                {new Date(transaction.created_at).toLocaleString()}
              </span>
            </div>
            <div className="detail-item">
              <span className="label">Day of Week:</span>
              <span className="value">
                {new Date(transaction.timestamp).toLocaleDateString('en-US', { weekday: 'long' })}
              </span>
            </div>
            <div className="detail-item">
              <span className="label">Time of Day:</span>
              <span className="value">
                {new Date(transaction.timestamp).toLocaleTimeString()}
              </span>
            </div>
          </div>

          {/* Customer Information */}
          {customer && (
            <div className="detail-card">
              <h3>Customer Information</h3>
              <div className="customer-info">
                <div className="customer-avatar">
                  {customer.first_name[0]}{customer.last_name}
                </div>
                <div className="customer-details">
                  <h4>{customer.first_name} {customer.last_name}</h4>
                  <p>ID: {customer.customer_id}</p>
                </div>
              </div>
              <div className="detail-item">
                <span className="label">Email:</span>
                <span className="value">{customer.email}</span>
              </div>
              <div className="detail-item">
                <span className="label">Phone:</span>
                <span className="value">{customer.phone}</span>
              </div>
              <button 
                onClick={() => navigate(`/customers/${customer.customer_id}`)}
                className="view-customer-btn"
              >
                View Customer Profile
              </button>
            </div>
          )}

          {/* Fraud Analysis */}
          <div className="detail-card fraud-analysis">
            <h3>Fraud Analysis</h3>
            <div className="fraud-status">
              <div className={`fraud-indicator ${transaction.is_fraud ? 'fraud' : 'normal'}`}>
                {transaction.is_fraud ? (
                  <span>🚨 FRAUDULENT TRANSACTION</span>
                ) : (
                  <span>✅ LEGITIMATE TRANSACTION</span>
                )}
              </div>
            </div>
            
            {transaction.is_fraud && (
              <div className="fraud-details">
                <h4>Fraud Indicators:</h4>
                <ul>
                  <li>Unusual transaction amount</li>
                  <li>Suspicious merchant activity</li>
                  <li>Geographic location mismatch</li>
                  <li>Time-based anomalies</li>
                </ul>
              </div>
            )}
            
            <div className="risk-metrics">
              <div className="metric">
                <span className="label">Risk Score:</span>
                <span className={`value ${transaction.is_fraud ? 'high' : 'low'}`}>
                  {transaction.is_fraud ? 'HIGH (85%)' : 'LOW (15%)'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="action-buttons">
          <button onClick={() => navigate('/transactions')} className="secondary-btn">
            Back to All Transactions
          </button>
          {customer && (
            <button 
              onClick={() => navigate(`/customers/${customer.customer_id}`)} 
              className="primary-btn"
            >
              View Customer Details
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default TransactionDetail;
