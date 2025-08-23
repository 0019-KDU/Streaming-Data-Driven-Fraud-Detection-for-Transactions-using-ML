import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  InputAdornment,
  Grid,
  Avatar,
  Chip,
  Button,
  IconButton,
  Pagination,
  Skeleton,
  Alert,
} from '@mui/material';
import {
  Search,
  Visibility,
  PersonAdd,
  FilterList,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import api from '../../services/api';

const CustomerCard = ({ customer, onViewDetails }) => (
  <Card sx={{ height: '100%', cursor: 'pointer', transition: 'all 0.2s', '&:hover': { transform: 'translateY(-4px)', boxShadow: 4 } }}>
    <CardContent sx={{ p: 3 }}>
      <Box display="flex" alignItems="center" gap={2} mb={2}>
        <Avatar 
          sx={{ 
            width: 56, 
            height: 56, 
            bgcolor: 'primary.main',
            fontSize: '1.25rem',
            fontWeight: 600
          }}
        >
          {customer.first_name[0]}{customer.last_name}
        </Avatar>
        <Box flex={1}>
          <Typography variant="h6" fontWeight={600} gutterBottom>
            {customer.first_name} {customer.last_name}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            ID: {customer.customer_id}
          </Typography>
        </Box>
        <Chip 
          label={customer.country_code} 
          size="small" 
          sx={{ fontWeight: 600 }}
        />
      </Box>
      
      <Box mb={2}>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          <strong>Email:</strong> {customer.email}
        </Typography>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          <strong>Phone:</strong> {customer.phone}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          <strong>DOB:</strong> {new Date(customer.date_of_birth).toLocaleDateString()}
        </Typography>
      </Box>

      <Button
        fullWidth
        variant="contained"
        startIcon={<Visibility />}
        onClick={() => onViewDetails(customer.customer_id)}
        sx={{ mt: 'auto' }}
      >
        View Details
      </Button>
    </CardContent>
  </Card>
);

const Customers = () => {
  const navigate = useNavigate();
  const [customers, setCustomers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [error, setError] = useState(null);
  const customersPerPage = 12;

  useEffect(() => {
    fetchCustomers();
  }, []);

  const fetchCustomers = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await api.get('/customers?limit=100');
      setCustomers(response.data);
    } catch (error) {
      setError('Failed to fetch customers. Please try again.');
      console.error('Error fetching customers:', error);
    } finally {
      setLoading(false);
    }
  };

  const filteredCustomers = customers.filter(customer =>
    `${customer.first_name} ${customer.last_name}`
      .toLowerCase()
      .includes(searchTerm.toLowerCase()) ||
    customer.email.toLowerCase().includes(searchTerm.toLowerCase()) ||
    customer.customer_id.toString().includes(searchTerm)
  );

  // Pagination
  const indexOfLastCustomer = currentPage * customersPerPage;
  const indexOfFirstCustomer = indexOfLastCustomer - customersPerPage;
  const currentCustomers = filteredCustomers.slice(indexOfFirstCustomer, indexOfLastCustomer);
  const totalPages = Math.ceil(filteredCustomers.length / customersPerPage);

  const goToCustomerDetail = (customerId) => {
    navigate(`/customers/${customerId}`);
  };

  if (error) {
    return (
      <Box p={3}>
        <Alert severity="error" action={
          <Button color="inherit" size="small" onClick={fetchCustomers}>
            Retry
          </Button>
        }>
          {error}
        </Alert>
      </Box>
    );
  }

  return (
    <Box p={3}>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Box>
          <Typography variant="h4" fontWeight={700} gutterBottom>
            Customer Management
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Manage and view customer information
          </Typography>
        </Box>
        <Button
          variant="contained"
          startIcon={<PersonAdd />}
          size="large"
        >
          Add Customer
        </Button>
      </Box>

      {/* Search and Stats */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box display="flex" gap={2} alignItems="center" mb={2}>
            <TextField
              placeholder="Search customers by name, email, or ID..."
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value);
                setCurrentPage(1);
              }}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Search />
                  </InputAdornment>
                ),
              }}
              sx={{ flex: 1, maxWidth: 400 }}
            />
            <IconButton>
              <FilterList />
            </IconButton>
          </Box>
          
          <Box display="flex" gap={4}>
            <Box>
              <Typography variant="h6" fontWeight={600}>
                {customers.length.toLocaleString()}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Total Customers
              </Typography>
            </Box>
            <Box>
              <Typography variant="h6" fontWeight={600}>
                {filteredCustomers.length.toLocaleString()}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Filtered Results
              </Typography>
            </Box>
          </Box>
        </CardContent>
      </Card>

      {/* Customer Grid */}
      <Grid container spacing={3}>
        {loading ? (
          Array.from(new Array(12)).map((_, index) => (
            <Grid item xs={12} sm={6} md={4} lg={3} key={index}>
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center" gap={2} mb={2}>
                    <Skeleton variant="circular" width={56} height={56} />
                    <Box flex={1}>
                      <Skeleton variant="text" width="80%" />
                      <Skeleton variant="text" width="60%" />
                    </Box>
                  </Box>
                  <Skeleton variant="text" />
                  <Skeleton variant="text" />
                  <Skeleton variant="text" />
                  <Skeleton variant="rectangular" width="100%" height={36} sx={{ mt: 2 }} />
                </CardContent>
              </Card>
            </Grid>
          ))
        ) : currentCustomers.length === 0 ? (
          <Grid item xs={12}>
            <Box textAlign="center" py={6}>
              <Typography variant="h6" color="text.secondary" gutterBottom>
                No customers found
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {searchTerm ? 'Try adjusting your search terms' : 'No customers have been added yet'}
              </Typography>
            </Box>
          </Grid>
        ) : (
          currentCustomers.map((customer) => (
            <Grid item xs={12} sm={6} md={4} lg={3} key={customer.customer_id}>
              <CustomerCard
                customer={customer}
                onViewDetails={goToCustomerDetail}
              />
            </Grid>
          ))
        )}
      </Grid>

      {/* Pagination */}
      {totalPages > 1 && (
        <Box display="flex" justifyContent="center" mt={4}>
          <Pagination
            count={totalPages}
            page={currentPage}
            onChange={(_, page) => setCurrentPage(page)}
            color="primary"
            size="large"
          />
        </Box>
      )}
    </Box>
  );
};

export default Customers;
