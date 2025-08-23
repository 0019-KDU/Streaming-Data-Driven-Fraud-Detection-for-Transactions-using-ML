import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Header from '../../components/common/Header';
import api from '../../services/api';
import './Transactions.css';

const Transactions = () => {
  const navigate = useNavigate();
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all'); // all, normal, fraud
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [transactionsPerPage] = useState(10);

  useEffect(() => {
    fetchTransactions();
  }, []);

  const fetchTransactions = async () => {
    try {
      const response = await api.get('/transactions');
      setTransactions(response.data);
    } catch (error) {
      console.error('Error fetching transactions:', error);
    } finally {
      setLoading(false);
    }
  };

  const filteredTransactions = transactions.filter(transaction => {
    const matchesFilter = 
      filter === 'all' || 
      (filter === 'fraud' && transaction.is_fraud) ||
      (filter === 'normal' && !transaction.is_fraud);
    
    const matchesSearch = 
      transaction.transaction_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      transaction.merchant.toLowerCase().includes(searchTerm.toLowerCase());
    
    return matchesFilter && matchesSearch;
  });

  // Pagination
  const indexOfLastTransaction = currentPage * transactionsPerPage;
  const indexOfFirstTransaction = indexOfLastTransaction - transactionsPerPage;
  const currentTransactions = filteredTransactions.slice(indexOfFirstTransaction, indexOfLastTransaction);
  const totalPages = Math.ceil(filteredTransactions.length / transactionsPerPage);

  const goToTransactionDetail = (transactionId) => {
    navigate(`/transactions/${transactionId}`);
  };

  if (loading) {
    return <div className="loading">Loading transactions...</div>;
  }

  return (
    <div className="transactions-page">
      <Header />
      <div className="transactions-content">
        <div className="transactions-header">
          <h2>Transaction Management</h2>
          
          <div className="controls-section">
            <div className="search-filter">
              <input
                type="text"
                placeholder="Search transactions..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="search-input"
              />
              
              <select 
                value={filter} 
                onChange={(e) => setFilter(e.target.value)}
                className="filter-select"
              >
                <option value="all">All Transactions</option>
                <option value="normal">Normal Only</option>
                <option value="fraud">Fraud Only</option>
              </select>
            </div>
          </div>
        </div>

        <div className="transactions-stats">
          <div className="stat-item">
            <span className="stat-label">Total:</span>
            <span className="stat-value">{transactions.length}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Normal:</span>
            <span className="stat-value">{transactions.filter(t => !t.is_fraud).length}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Fraud:</span>
            <span className="stat-value fraud">{transactions.filter(t => t.is_fraud).length}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Filtered:</span>
            <span className="stat-value">{filteredTransactions.length}</span>
          </div>
        </div>

        <div className="transactions-table">
          <div className="table-header">
            <div className="col-transaction-id">Transaction ID</div>
            <div className="col-customer">Customer ID</div>
            <div className="col-merchant">Merchant</div>
            <div className="col-amount">Amount</div>
            <div className="col-date">Date</div>
            <div className="col-status">Status</div>
            <div className="col-actions">Actions</div>
          </div>
          
          {currentTransactions.map(transaction => (
            <div key={transaction.transaction_id} className="table-row">
              <div className="col-transaction-id">
                {transaction.transaction_id.substring(0, 8)}...
              </div>
              <div className="col-customer">{transaction.user_id}</div>
              <div className="col-merchant">{transaction.merchant}</div>
              <div className="col-amount">${transaction.amount.toLocaleString()}</div>
              <div className="col-date">
                {new Date(transaction.timestamp).toLocaleDateString()}
              </div>
              <div className="col-status">
                <span className={`status-badge ${transaction.is_fraud ? 'fraud' : 'normal'}`}>
                  {transaction.is_fraud ? 'Fraud' : 'Normal'}
                </span>
              </div>
              <div className="col-actions">
                <button 
                  onClick={() => goToTransactionDetail(transaction.transaction_id)}
                  className="detail-btn"
                >
                  Details
                </button>
              </div>
            </div>
          ))}
        </div>

        {/* Pagination */}
        <div className="pagination">
          <button 
            onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
            disabled={currentPage === 1}
            className="page-btn"
          >
            Previous
          </button>
          
          <span className="page-info">
            Page {currentPage} of {totalPages}
          </span>
          
          <button 
            onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
            disabled={currentPage === totalPages}
            className="page-btn"
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
};

export default Transactions;
