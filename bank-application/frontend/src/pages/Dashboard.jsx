import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Avatar,
  IconButton,
  Chip,
  LinearProgress,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  Divider,
  Button,
  Paper,
  CircularProgress,
} from '@mui/material';
import {
  TrendingUp,
  People,
  Receipt,
  Warning,
  AccountBalance,
  Refresh,
  ArrowForward,
  TrendingDown,
} from '@mui/icons-material';
import { useAuth } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';
import api from '../services/api';

const StatCard = ({ title, value, icon: Icon, color, subtitle, trend, loading }) => (
  <Card sx={{ height: '100%', position: 'relative', overflow: 'visible' }}>
    {loading && <LinearProgress sx={{ position: 'absolute', top: 0, left: 0, right: 0 }} />}
    <CardContent sx={{ p: 3 }}>
      <Box display="flex" alignItems="center" justifyContent="space-between">
        <Box flex={1}>
          <Typography variant="h3" component="h2" sx={{ fontWeight: 700, mb: 1, color: color }}>
            {loading ? '...' : (typeof value === 'number' ? value.toLocaleString() : value)}
          </Typography>
          <Typography variant="h6" color="text.secondary" gutterBottom>
            {title}
          </Typography>
          {subtitle && (
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              {subtitle}
            </Typography>
          )}
          {trend && (
            <Chip 
              label={trend} 
              size="small" 
              color={trend.includes('+') ? 'success' : trend.includes('-') ? 'error' : 'default'}
              sx={{ fontWeight: 600 }}
            />
          )}
        </Box>
        <Avatar 
          sx={{ 
            bgcolor: color, 
            width: 70, 
            height: 70,
            boxShadow: `0 8px 16px ${color}40`,
          }}
        >
          <Icon sx={{ fontSize: 32 }} />
        </Avatar>
      </Box>
    </CardContent>
  </Card>
);

const Dashboard = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [stats, setStats] = useState({
    totalCustomers: 0,
    totalTransactions: 0,
    totalAmount: 0,
    fraudTransactions: 0
  });
  const [recentCustomers, setRecentCustomers] = useState([]);
  const [recentTransactions, setRecentTransactions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(new Date());

  const fetchDashboardData = async () => {
    setLoading(true);
    try {
      const [customersRes, transactionsRes, dashboardRes] = await Promise.all([
        api.get('/customers?limit=5'),
        api.get('/transactions?limit=5'),
        api.get('/dashboard/stats')
      ]);

      setRecentCustomers(customersRes.data);
      setRecentTransactions(transactionsRes.data);
      setStats(dashboardRes.data);
      setLastUpdated(new Date());
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fraudRate = stats.totalTransactions > 0 
    ? ((stats.fraudTransactions / stats.totalTransactions) * 100).toFixed(2)
    : 0;

  const avgTransactionAmount = stats.totalTransactions > 0
    ? (stats.totalAmount / stats.totalTransactions).toFixed(2)
    : 0;

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={4}>
        <Box>
          <Typography variant="h4" component="h1" fontWeight={700} gutterBottom>
            Welcome back, {user?.full_name}!
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Here's what's happening with your bank today • Last updated: {lastUpdated.toLocaleTimeString()}
          </Typography>
        </Box>
        <IconButton 
          onClick={fetchDashboardData} 
          color="primary"
          sx={{ 
            bgcolor: 'primary.main', 
            color: 'white',
            '&:hover': { bgcolor: 'primary.dark' }
          }}
        >
          <Refresh />
        </IconButton>
      </Box>

      {/* Stats Cards */}
      <Grid container spacing={3} mb={4}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Total Customers"
            value={stats.totalCustomers}
            icon={People}
            color="#1976d2"
            subtitle="Active accounts"
            trend="+12% this month"
            loading={loading}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Total Transactions"
            value={stats.totalTransactions}
            icon={Receipt}
            color="#2e7d32"
            subtitle="All time"
            trend="+5.3% today"
            loading={loading}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Total Amount"
            value={`$${stats.totalAmount.toLocaleString()}`}
            icon={AccountBalance}
            color="#ed6c02"
            subtitle={`Avg: $${avgTransactionAmount}`}
            trend="+8.1% this week"
            loading={loading}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Fraud Alerts"
            value={stats.fraudTransactions}
            icon={Warning}
            color="#d32f2f"
            subtitle={`${fraudRate}% fraud rate`}
            trend={parseFloat(fraudRate) < 2 ? "Within normal range" : "Above threshold"}
            loading={loading}
          />
        </Grid>
      </Grid>

      {/* Content Grid */}
      <Grid container spacing={3}>
        {/* Recent Customers */}
        <Grid item xs={12} lg={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent sx={{ p: 3 }}>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                <Typography variant="h6" fontWeight={600}>
                  Recent Customers
                </Typography>
                <Button 
                  endIcon={<ArrowForward />}
                  onClick={() => navigate('/customers')}
                  size="small"
                >
                  View All
                </Button>
              </Box>
              
              {loading ? (
                <Box display="flex" justifyContent="center" p={3}>
                  <CircularProgress />
                </Box>
              ) : (
                <List>
                  {recentCustomers.map((customer, index) => (
                    <React.Fragment key={customer.customer_id}>
                      <ListItem 
                        sx={{ px: 0, cursor: 'pointer' }}
                        onClick={() => navigate(`/customers/${customer.customer_id}`)}
                      >
                        <ListItemAvatar>
                          <Avatar sx={{ bgcolor: 'primary.main' }}>
                            {customer.first_name[0]}{customer.last_name}
                          </Avatar>
                        </ListItemAvatar>
                        <ListItemText 
                          primary={`${customer.first_name} ${customer.last_name}`}
                          secondary={customer.email}
                          primaryTypographyProps={{ fontWeight: 500 }}
                        />
                        <Chip label={customer.country_code} size="small" />
                      </ListItem>
                      {index < recentCustomers.length - 1 && <Divider />}
                    </React.Fragment>
                  ))}
                </List>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Transactions */}
        <Grid item xs={12} lg={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent sx={{ p: 3 }}>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                <Typography variant="h6" fontWeight={600}>
                  Recent Transactions
                </Typography>
                <Button 
                  endIcon={<ArrowForward />}
                  onClick={() => navigate('/transactions')}
                  size="small"
                >
                  View All
                </Button>
              </Box>
              
              {loading ? (
                <Box display="flex" justifyContent="center" p={3}>
                  <CircularProgress />
                </Box>
              ) : (
                <List>
                  {recentTransactions.map((transaction, index) => (
                    <React.Fragment key={transaction.transaction_id}>
                      <ListItem 
                        sx={{ px: 0, cursor: 'pointer' }}
                        onClick={() => navigate(`/transactions/${transaction.transaction_id}`)}
                      >
                        <ListItemAvatar>
                          <Avatar sx={{ bgcolor: transaction.is_fraud ? 'error.main' : 'success.main' }}>
                            {transaction.is_fraud ? <Warning /> : <Receipt />}
                          </Avatar>
                        </ListItemAvatar>
                        <ListItemText 
                          primary={transaction.merchant}
                          secondary={`${transaction.transaction_id.substring(0, 8)}... • ${new Date(transaction.timestamp).toLocaleDateString()}`}
                          primaryTypographyProps={{ fontWeight: 500 }}
                        />
                        <Box textAlign="right">
                          <Typography variant="h6" fontWeight={600} color={transaction.amount > 5000 ? 'error.main' : 'success.main'}>
                            ${transaction.amount.toLocaleString()}
                          </Typography>
                          <Chip 
                            label={transaction.is_fraud ? 'Fraud' : 'Normal'} 
                            color={transaction.is_fraud ? 'error' : 'success'}
                            size="small"
                          />
                        </Box>
                      </ListItem>
                      {index < recentTransactions.length - 1 && <Divider />}
                    </React.Fragment>
                  ))}
                </List>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Quick Actions */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3, mt: 2 }}>
            <Typography variant="h6" fontWeight={600} mb={2}>
              Quick Actions
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6} md={3}>
                <Button
                  fullWidth
                  variant="outlined"
                  startIcon={<People />}
                  onClick={() => navigate('/customers')}
                  sx={{ py: 1.5 }}
                >
                  Manage Customers
                </Button>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Button
                  fullWidth
                  variant="outlined"
                  startIcon={<Receipt />}
                  onClick={() => navigate('/transactions')}
                  sx={{ py: 1.5 }}
                >
                  View Transactions
                </Button>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Button
                  fullWidth
                  variant="outlined"
                  startIcon={<Warning />}
                  color="error"
                  sx={{ py: 1.5 }}
                >
                  Fraud Report
                </Button>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Button
                  fullWidth
                  variant="outlined"
                  startIcon={<TrendingUp />}
                  sx={{ py: 1.5 }}
                >
                  Analytics
                </Button>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;
