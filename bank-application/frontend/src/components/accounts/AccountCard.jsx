import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  Avatar,
  Divider,
  LinearProgress,
} from '@mui/material';
import {
  AccountBalance,
  CreditCard,
  TrendingUp,
  Business,
  School,
  Elderly,
  People,
  Savings,
} from '@mui/icons-material';

const getAccountTypeIcon = (type) => {
  const icons = {
    'savings': <Savings />,
    'checking': <AccountBalance />,
    'business': <Business />,
    'student': <School />,
    'senior': <Elderly />,
    'joint': <People />
  };
  return icons[type] || <AccountBalance />;
};

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

const AccountCard = ({ account }) => {
  const linkedServices = account.linked_services ? JSON.parse(account.linked_services) : [];
  const accountTypeColor = getAccountTypeColor(account.account_type);
  const balancePercentage = account.minimum_balance > 0 
    ? Math.min((parseFloat(account.balance) / parseFloat(account.minimum_balance)) * 100, 100)
    : 100;
  
  return (
    <Card 
      sx={{ 
        height: '100%', 
        borderRadius: 3,
        background: getAccountTypeGradient(account.account_type),
        color: 'white',
        position: 'relative',
        overflow: 'visible',
        '&:hover': {
          transform: 'translateY(-4px)',
          boxShadow: `0 12px 24px ${accountTypeColor}40`,
        },
        transition: 'all 0.3s ease'
      }}
    >
      <CardContent sx={{ p: 3, position: 'relative' }}>
        {/* Account Type Badge */}
        <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
          <Box display="flex" alignItems="center" gap={1}>
            <Avatar sx={{ bgcolor: 'rgba(255,255,255,0.2)', width: 45, height: 45 }}>
              {getAccountTypeIcon(account.account_type)}
            </Avatar>
            <Box>
              <Typography variant="h6" fontWeight={700} sx={{ textTransform: 'uppercase', letterSpacing: 1 }}>
                {account.account_type.replace('_', ' ')}
              </Typography>
              <Typography variant="body2" sx={{ opacity: 0.9, fontFamily: 'monospace', fontSize: '0.85rem' }}>
                {account.account_number}
              </Typography>
            </Box>
          </Box>
          <Chip 
            label={account.account_status.toUpperCase()} 
            sx={{
              bgcolor: account.account_status === 'active' ? 'rgba(76, 175, 80, 0.9)' : 'rgba(158, 158, 158, 0.9)',
              color: 'white',
              fontWeight: 600,
              fontSize: '0.7rem'
            }}
            size="small"
          />
        </Box>

        {/* Balance Display */}
        <Box mb={3}>
          <Typography variant="body2" sx={{ opacity: 0.8, mb: 0.5 }}>
            Current Balance
          </Typography>
          <Typography variant="h3" fontWeight={800} sx={{ mb: 1 }}>
            {formatCurrency(account.balance)}
          </Typography>
          
          {/* Balance Progress Bar */}
          {account.minimum_balance > 0 && (
            <Box>
              <Typography variant="body2" sx={{ opacity: 0.8, fontSize: '0.75rem', mb: 0.5 }}>
                Minimum Balance: {formatCurrency(account.minimum_balance)}
              </Typography>
              <LinearProgress 
                variant="determinate" 
                value={balancePercentage}
                sx={{
                  height: 6,
                  borderRadius: 3,
                  backgroundColor: 'rgba(255,255,255,0.2)',
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: balancePercentage > 50 ? '#4caf50' : '#ff9800',
                    borderRadius: 3,
                  }
                }}
              />
            </Box>
          )}
        </Box>

        <Divider sx={{ bgcolor: 'rgba(255,255,255,0.2)', my: 2 }} />

        {/* Account Details */}
        <Box mb={2}>
          <Box display="flex" justifyContent="space-between" mb={1}>
            <Typography variant="body2" sx={{ opacity: 0.8 }}>Created</Typography>
            <Typography variant="body2" fontWeight={500}>
              {new Date(account.created_at).toLocaleDateString('en-US', {
                month: 'short',
                day: 'numeric',
                year: 'numeric'
              })}
            </Typography>
          </Box>
          
          {account.interest_rate > 0 && (
            <Box display="flex" justifyContent="space-between" mb={1}>
              <Typography variant="body2" sx={{ opacity: 0.8 }}>Interest Rate</Typography>
              <Typography variant="body2" fontWeight={500}>
                {(parseFloat(account.interest_rate) * 100).toFixed(2)}% APY
              </Typography>
            </Box>
          )}
          
          {account.overdraft_limit > 0 && (
            <Box display="flex" justifyContent="space-between" mb={1}>
              <Typography variant="body2" sx={{ opacity: 0.8 }}>Overdraft Limit</Typography>
              <Typography variant="body2" fontWeight={500}>
                {formatCurrency(account.overdraft_limit)}
              </Typography>
            </Box>
          )}
        </Box>

        {/* Linked Services */}
        {linkedServices.length > 0 && (
          <Box>
            <Typography variant="body2" sx={{ opacity: 0.8, mb: 1, fontSize: '0.8rem' }}>
              Linked Services ({linkedServices.length})
            </Typography>
            <Box display="flex" flexWrap="wrap" gap={0.5}>
              {linkedServices.slice(0, 4).map((service, index) => (
                <Chip
                  key={index}
                  label={service.replace('_', ' ').toUpperCase()}
                  size="small"
                  sx={{
                    bgcolor: 'rgba(255,255,255,0.15)',
                    color: 'white',
                    fontSize: '0.65rem',
                    height: 22,
                    '& .MuiChip-label': { px: 1 }
                  }}
                />
              ))}
              {linkedServices.length > 4 && (
                <Chip
                  label={`+${linkedServices.length - 4} more`}
                  size="small"
                  sx={{
                    bgcolor: 'rgba(255,255,255,0.1)',
                    color: 'white',
                    fontSize: '0.65rem',
                    height: 22,
                  }}
                />
              )}
            </Box>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default AccountCard;
