import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import Header from '../../components/common/Header';
import api from '../../services/api';
import './CustomerDetail.css';

// Account Card Component (inline for completeness)
const AccountCard = ({ account }) => {
  const getAccountTypeColor = (type) => {
    const colors = {
      'savings': '#2e7d32',
      'checking': '#1976d2', 
      'business': '#ed6c02',
      'student': '#9c27b0',
      'senior': '#795548',
      'joint': '#00796b'
    };
    return colors[type] || '#1976d2';
  };

  const getAccountTypeGradient = (type) => {
    const gradients = {
      'savings': 'linear-gradient(135deg, #2e7d32 0%, #4caf50 100%)',
      'checking': 'linear-gradient(135deg, #1976d2 0%, #42a5f5 100%)',
      'business': 'linear-gradient(135deg, #ed6c02 0%, #ff9800 100%)',
      'student': 'linear-gradient(135deg, #9c27b0 0%, #ba68c8 100%)',
      'senior': 'linear-gradient(135deg, #795548 0%, #8d6e63 100%)',
      'joint': 'linear-gradient(135deg, #00796b 0%, #26a69a 100%)'
    };
    return gradients[type] || gradients.checking;
  };

  const formatCurrency = (amount) => {
    return parseFloat(amount).toLocaleString('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    });
  };

  const linkedServices = account.linked_services ? JSON.parse(account.linked_services) : [];
  const accountTypeColor = getAccountTypeColor(account.account_type);

  return (
    <div 
      style={{ 
        height: '100%', 
        borderRadius: '12px',
        background: getAccountTypeGradient(account.account_type),
        color: 'white',
        position: 'relative',
        overflow: 'visible',
        transition: 'all 0.3s ease',
        cursor: 'pointer',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.transform = 'translateY(-4px)';
        e.currentTarget.style.boxShadow = `0 12px 24px ${accountTypeColor}40`;
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.transform = 'translateY(0)';
        e.currentTarget.style.boxShadow = 'none';
      }}
    >
      <div style={{ padding: '24px', position: 'relative' }}>
        {/* Account Type Badge */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{
              backgroundColor: 'rgba(255,255,255,0.2)',
              width: '45px',
              height: '45px',
              borderRadius: '50%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '20px'
            }}>
              💳
            </div>
            <div>
              <div style={{ 
                fontWeight: 700, 
                textTransform: 'uppercase', 
                letterSpacing: '1px',
                fontSize: '18px',
                marginBottom: '4px'
              }}>
                {account.account_type.replace('_', ' ')}
              </div>
              <div style={{ 
                opacity: 0.9, 
                fontFamily: 'monospace', 
                fontSize: '14px' 
              }}>
                {account.account_number}
              </div>
            </div>
          </div>
          <span style={{
            backgroundColor: account.account_status === 'active' ? 'rgba(76, 175, 80, 0.9)' : 'rgba(158, 158, 158, 0.9)',
            color: 'white',
            fontWeight: 600,
            fontSize: '12px',
            padding: '4px 8px',
            borderRadius: '4px'
          }}>
            {account.account_status.toUpperCase()}
          </span>
        </div>

        {/* Balance Display */}
        <div style={{ marginBottom: '24px' }}>
          <div style={{ opacity: 0.8, marginBottom: '4px', fontSize: '14px' }}>
            Current Balance
          </div>
          <div style={{ fontWeight: 800, fontSize: '28px', marginBottom: '8px' }}>
            {formatCurrency(account.balance)}
          </div>
        </div>

        <div style={{ height: '1px', backgroundColor: 'rgba(255,255,255,0.2)', margin: '16px 0' }} />

        {/* Account Details */}
        <div style={{ marginBottom: '16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ opacity: 0.8, fontSize: '14px' }}>Created</span>
            <span style={{ fontWeight: 500, fontSize: '14px' }}>
              {new Date(account.created_at).toLocaleDateString('en-US', {
                month: 'short',
                day: 'numeric',
                year: 'numeric'
              })}
            </span>
          </div>
          
          {account.interest_rate > 0 && (
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <span style={{ opacity: 0.8, fontSize: '14px' }}>Interest Rate</span>
              <span style={{ fontWeight: 500, fontSize: '14px' }}>
                {(parseFloat(account.interest_rate) * 100).toFixed(2)}% APY
              </span>
            </div>
          )}
          
          {account.overdraft_limit > 0 && (
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <span style={{ opacity: 0.8, fontSize: '14px' }}>Overdraft Limit</span>
              <span style={{ fontWeight: 500, fontSize: '14px' }}>
                {formatCurrency(account.overdraft_limit)}
              </span>
            </div>
          )}
        </div>

        {/* Linked Services */}
        {linkedServices.length > 0 && (
          <div>
            <div style={{ opacity: 0.8, marginBottom: '8px', fontSize: '13px' }}>
              Linked Services ({linkedServices.length})
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
              {linkedServices.slice(0, 4).map((service, index) => (
                <span
                  key={index}
                  style={{
                    backgroundColor: 'rgba(255,255,255,0.15)',
                    color: 'white',
                    fontSize: '11px',
                    height: '22px',
                    padding: '0 8px',
                    borderRadius: '11px',
                    display: 'flex',
                    alignItems: 'center'
                  }}
                >
                  {service.replace('_', ' ').toUpperCase()}
                </span>
              ))}
              {linkedServices.length > 4 && (
                <span
                  style={{
                    backgroundColor: 'rgba(255,255,255,0.1)',
                    color: 'white',
                    fontSize: '11px',
                    height: '22px',
                    padding: '0 8px',
                    borderRadius: '11px',
                    display: 'flex',
                    alignItems: 'center'
                  }}
                >
                  +{linkedServices.length - 4} more
                </span>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const CustomerDetail = () => {
  const { customerId } = useParams();
  const navigate = useNavigate();
  const [customer, setCustomer] = useState(null);
  const [transactions, setTransactions] = useState([]);
  const [accounts, setAccounts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('details');
  const [transactionStats, setTransactionStats] = useState({
    total: 0,
    totalAmount: 0,
    fraudCount: 0
  });
  const [accountStats, setAccountStats] = useState({
    totalAccounts: 0,
    totalBalance: 0,
    averageBalance: 0
  });

  useEffect(() => {
    fetchCustomerData();
  }, [customerId]);

  const fetchCustomerData = async () => {
    try {
      const [customerRes, transactionsRes, accountsRes] = await Promise.all([
        api.get(`/api/v1/customers/${customerId}`),
        api.get(`/api/v1/customers/${customerId}/transactions`),
        api.get(`/api/v1/accounts/customer/${customerId}`)
      ]);

      setCustomer(customerRes.data);
      setTransactions(transactionsRes.data);
      setAccounts(accountsRes.data);
      
      // Calculate transaction stats
      const stats = transactionsRes.data.reduce((acc, transaction) => ({
        total: acc.total + 1,
        totalAmount: acc.totalAmount + transaction.amount,
        fraudCount: acc.fraudCount + (transaction.is_fraud ? 1 : 0)
      }), { total: 0, totalAmount: 0, fraudCount: 0 });
      
      setTransactionStats(stats);

      // Calculate account stats
      const accountStatsCalc = accountsRes.data.reduce((acc, account) => {
        const balance = parseFloat(account.balance);
        return {
          totalAccounts: acc.totalAccounts + 1,
          totalBalance: acc.totalBalance + balance,
          averageBalance: 0 // Will calculate after
        };
      }, { totalAccounts: 0, totalBalance: 0, averageBalance: 0 });
      
      accountStatsCalc.averageBalance = accountStatsCalc.totalAccounts > 0 
        ? accountStatsCalc.totalBalance / accountStatsCalc.totalAccounts 
        : 0;
      
      setAccountStats(accountStatsCalc);
      
    } catch (error) {
      console.error('Error fetching customer data:', error);
    } finally {
      setLoading(false);
    }
  };

  const goToTransactionDetail = (transactionId) => {
    navigate(`/transactions/${transactionId}`);
  };

  if (loading) {
    return <div className="loading">Loading customer details...</div>;
  }

  if (!customer) {
    return <div className="error">Customer not found</div>;
  }

  return (
    <div className="customer-detail-page">
      <Header />
      <div className="customer-detail-content">
        {/* Customer Header */}
        <div className="customer-detail-header">
          <button onClick={() => navigate('/customers')} className="back-btn">
            ← Back to Customers
          </button>
          <div className="customer-title">
            <div className="customer-avatar-large">
              {customer.first_name[0]}{customer.last_name}
            </div>
            <div className="customer-title-info">
              <h1>{customer.first_name} {customer.last_name}</h1>
              <p className="customer-id">Customer ID: {customer.customer_id}</p>
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="tabs">
          <button 
            className={`tab ${activeTab === 'details' ? 'active' : ''}`}
            onClick={() => setActiveTab('details')}
          >
            Customer Details
          </button>
          <button 
            className={`tab ${activeTab === 'accounts' ? 'active' : ''}`}
            onClick={() => setActiveTab('accounts')}
          >
            Accounts ({accounts.length})
          </button>
          <button 
            className={`tab ${activeTab === 'transactions' ? 'active' : ''}`}
            onClick={() => setActiveTab('transactions')}
          >
            Transactions ({transactionStats.total})
          </button>
        </div>

        {/* Tab Content */}
        {activeTab === 'details' && (
          <div className="customer-details-section">
            <div className="details-grid">
              <div className="detail-card">
                <h3>Personal Information</h3>
                <div className="detail-item">
                  <span className="label">Full Name:</span>
                  <span className="value">{customer.first_name} {customer.last_name}</span>
                </div>
                <div className="detail-item">
                  <span className="label">Date of Birth:</span>
                  <span className="value">{new Date(customer.date_of_birth).toLocaleDateString()}</span>
                </div>
                <div className="detail-item">
                  <span className="label">National ID:</span>
                  <span className="value">{customer.national_id}</span>
                </div>
              </div>

              <div className="detail-card">
                <h3>Contact Information</h3>
                <div className="detail-item">
                  <span className="label">Email:</span>
                  <span className="value">{customer.email}</span>
                </div>
                <div className="detail-item">
                  <span className="label">Phone:</span>
                  <span className="value">{customer.phone}</span>
                </div>
                <div className="detail-item">
                  <span className="label">Address:</span>
                  <span className="value">{customer.address}</span>
                </div>
              </div>

              <div className="detail-card">
                <h3>Account Summary</h3>
                <div className="detail-item">
                  <span className="label">Total Accounts:</span>
                  <span className="value">{accountStats.totalAccounts}</span>
                </div>
                <div className="detail-item">
                  <span className="label">Total Balance:</span>
                  <span className="value">${accountStats.totalBalance.toLocaleString('en-US', { minimumFractionDigits: 2 })}</span>
                </div>
                <div className="detail-item">
                  <span className="label">Average Balance:</span>
                  <span className="value">${accountStats.averageBalance.toLocaleString('en-US', { minimumFractionDigits: 2 })}</span>
                </div>
              </div>

              <div className="detail-card">
                <h3>Transaction Summary</h3>
                <div className="detail-item">
                  <span className="label">Total Transactions:</span>
                  <span className="value">{transactionStats.total}</span>
                </div>
                <div className="detail-item">
                  <span className="label">Total Amount:</span>
                  <span className="value">${transactionStats.totalAmount.toLocaleString()}</span>
                </div>
                <div className="detail-item">
                  <span className="label">Fraud Alerts:</span>
                  <span className="value fraud-count">{transactionStats.fraudCount}</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'accounts' && (
          <div className="accounts-section">
            <div className="accounts-header" style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'center', 
              marginBottom: '2rem',
              padding: '1.5rem',
              backgroundColor: 'white',
              borderRadius: '12px',
              boxShadow: '0 4px 6px rgba(0, 0, 0, 0.05)'
            }}>
              <h3 style={{ margin: 0, color: '#2d3748', fontSize: '1.3rem', fontWeight: 600 }}>
                Bank Accounts
              </h3>
              <div className="account-stats" style={{ 
                display: 'flex', 
                gap: '2rem', 
                color: '#718096', 
                fontSize: '0.9rem' 
              }}>
                <span style={{ 
                  padding: '0.5rem 1rem', 
                  backgroundColor: '#f7fafc', 
                  borderRadius: '6px', 
                  fontWeight: 500 
                }}>
                  Total: {accountStats.totalAccounts}
                </span>
                <span style={{ 
                  padding: '0.5rem 1rem', 
                  backgroundColor: '#f7fafc', 
                  borderRadius: '6px', 
                  fontWeight: 500 
                }}>
                  Balance: ${accountStats.totalBalance.toLocaleString('en-US', { minimumFractionDigits: 2 })}
                </span>
                <span style={{ 
                  padding: '0.5rem 1rem', 
                  backgroundColor: '#f7fafc', 
                  borderRadius: '6px', 
                  fontWeight: 500 
                }}>
                  Avg: ${accountStats.averageBalance.toLocaleString('en-US', { minimumFractionDigits: 2 })}
                </span>
              </div>
            </div>

            <div className="accounts-grid" style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))', 
              gap: '2rem' 
            }}>
              {accounts.length === 0 ? (
                <div style={{ 
                  textAlign: 'center', 
                  padding: '4rem', 
                  color: '#718096',
                  backgroundColor: 'white',
                  borderRadius: '12px',
                  gridColumn: '1 / -1'
                }}>
                  <p style={{ fontSize: '1.1rem', marginBottom: '0.5rem' }}>No accounts found for this customer.</p>
                  <p style={{ fontSize: '0.9rem', margin: 0 }}>Accounts will be automatically generated when available.</p>
                </div>
              ) : (
                accounts.map(account => (
                  <AccountCard key={account.id} account={account} />
                ))
              )}
            </div>
          </div>
        )}

        {activeTab === 'transactions' && (
          <div className="transactions-section">
            <div className="transactions-header">
              <h3>Customer Transactions</h3>
              <div className="transaction-stats">
                <span>Total: {transactionStats.total}</span>
                <span>Amount: ${transactionStats.totalAmount.toLocaleString()}</span>
                <span>Fraud: {transactionStats.fraudCount}</span>
              </div>
            </div>

            <div className="transactions-list">
              {transactions.length === 0 ? (
                <div className="no-transactions">
                  <p>No transactions found for this customer.</p>
                </div>
              ) : (
                transactions.map(transaction => (
                  <div key={transaction.transaction_id} className="transaction-card">
                    <div className="transaction-main">
                      <div className="transaction-info">
                        <div className="transaction-id">
                          {transaction.transaction_id.substring(0, 12)}...
                        </div>
                        <div className="merchant">{transaction.merchant}</div>
                        <div className="transaction-date">
                          {new Date(transaction.timestamp).toLocaleDateString()}
                        </div>
                      </div>
                      
                      <div className="transaction-amount">
                        <span className="amount">${transaction.amount.toLocaleString()}</span>
                        <span className={`status ${transaction.is_fraud ? 'fraud' : 'normal'}`}>
                          {transaction.is_fraud ? '⚠️ Fraud' : '✅ Normal'}
                        </span>
                      </div>
                    </div>
                    
                    <button 
                      onClick={() => goToTransactionDetail(transaction.transaction_id)}
                      className="transaction-detail-btn"
                    >
                      View Details
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default CustomerDetail;
