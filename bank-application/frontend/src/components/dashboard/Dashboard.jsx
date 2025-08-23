import React, { useState, useEffect } from 'react';
import { 
  Card, 
  CardContent, 
  Typography, 
  Grid, 
  Box, 
  CircularProgress,
  IconButton,
  Chip,
  Avatar
} from '@mui/material';
import {
  TrendingUp,
  People,
  Receipt,
  Warning,
  AccountBalance,
  Refresh
} from '@mui/icons-material';
import api from '../../services/api';
import './Dashboard.css';

const StatCard = ({ title, value, icon: Icon, color, subtitle, trend }) => (
  <Card className="stat-card" elevation={3}>
    <CardContent>
      <Box display="flex" alignItems="center" justifyContent="space-between">
        <Box>
          <Typography variant="h4" component="h2" sx={{ fontWeight: 'bold', mb: 1 }}>
            {value.toLocaleString()}
          </Typography>
          <Typography variant="h6" color="text.secondary" gutterBottom>
            {title}
          </Typography>
          {subtitle && (
            <Typography variant="body2" color="text.secondary">
              {subtitle}
            </Typography>
          )}
          {trend && (
            <Chip 
              label={trend} 
              size="small" 
              color={trend.startsWith('+') ? 'success' : 'error'}
              sx={{ mt: 1 }}
            />
          )}
        </Box>
        <Avatar sx={{ bgcolor: color, width: 60, height: 60 }}>
          <Icon fontSize="large" />
        </Avatar>
      </Box>
    </CardContent>
  </Card>
);

const Dashboard = () => {
  const [stats, setStats] = useState({
    totalCustomers: 0,
    totalTransactions: 0,
    totalAmount: 0,
    fraudTransactions: 0
  });
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(new Date());

  const fetchDashboardData = async () => {
    setLoading(true);
    try {
      const response = await api.get('/api/v1/dashboard/stats');
      setStats(response.data);
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

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress size={60} />
      </Box>
    );
  }

  return (
    <Box className="dashboard-container">
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={4}>
        <Box>
          <Typography variant="h4" component="h1" gutterBottom>
            Banking Dashboard
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Last updated: {lastUpdated.toLocaleString()}
          </Typography>
        </Box>
        <IconButton onClick={fetchDashboardData} color="primary">
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
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Fraud Alerts"
            value={stats.fraudTransactions}
            icon={Warning}
            color="#d32f2f"
            subtitle={`${fraudRate}% fraud rate`}
            trend={fraudRate < 2 ? "Within normal range" : "Above threshold"}
          />
        </Grid>
      </Grid>

      {/* Quick Actions */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Quick Actions
              </Typography>
              <Box display="flex" flexDirection="column" gap={1}>
                <Chip 
                  label="View All Customers" 
                  color="primary" 
                  variant="outlined"
                  clickable
                  onClick={() => window.location.href = '/customers'}
                />
                <Chip 
                  label="View All Transactions" 
                  color="primary" 
                  variant="outlined"
                  clickable
                  onClick={() => window.location.href = '/transactions'}
                />
                <Chip 
                  label="Fraud Detection Report" 
                  color="error" 
                  variant="outlined"
                  clickable
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Status
              </Typography>
              <Box display="flex" flexDirection="column" gap={2}>
                <Box display="flex" justifyContent="space-between" alignItems="center">
                  <Typography variant="body2">Database</Typography>
                  <Chip label="Online" color="success" size="small" />
                </Box>
                <Box display="flex" justifyContent="space-between" alignItems="center">
                  <Typography variant="body2">Kafka Consumer</Typography>
                  <Chip label="Active" color="success" size="small" />
                </Box>
                <Box display="flex" justifyContent="space-between" alignItems="center">
                  <Typography variant="body2">Fraud Detection</Typography>
                  <Chip label="Running" color="success" size="small" />
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;
