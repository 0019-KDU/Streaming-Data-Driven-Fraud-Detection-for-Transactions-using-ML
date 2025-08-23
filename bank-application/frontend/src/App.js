import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { AuthProvider } from './context/AuthContext';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import Customers from './pages/Customers';
import CustomerDetail from './pages/Customers/CustomerDetail';
import Transactions from './pages/Transactions';
import TransactionDetail from './pages/Transactions/TransactionDetail';
import ProtectedRoute from './components/common/ProtectedRoute';
import Layout from './components/Layout/Layout';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
      light: '#42a5f5',
      dark: '#1565c0',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f7fa',
      paper: '#ffffff',
    },
    success: {
      main: '#2e7d32',
    },
    warning: {
      main: '#ed6c02',
    },
    error: {
      main: '#d32f2f',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
    },
    h5: {
      fontWeight: 500,
    },
    h6: {
      fontWeight: 500,
    },
  },
  shape: {
    borderRadius: 12,
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
          borderRadius: 12,
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          borderRadius: 8,
        },
      },
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AuthProvider>
        <Router>
          <div className="App">
            <Routes>
              <Route path="/login" element={<Login />} />
              <Route 
                path="/" 
                element={
                  <ProtectedRoute>
                    <Layout>
                      <Dashboard />
                    </Layout>
                  </ProtectedRoute>
                } 
              />
              <Route 
                path="/dashboard" 
                element={
                  <ProtectedRoute>
                    <Layout>
                      <Dashboard />
                    </Layout>
                  </ProtectedRoute>
                } 
              />
              <Route 
                path="/customers" 
                element={
                  <ProtectedRoute>
                    <Layout>
                      <Customers />
                    </Layout>
                  </ProtectedRoute>
                } 
              />
              <Route 
                path="/customers/:customerId" 
                element={
                  <ProtectedRoute>
                    <Layout>
                      <CustomerDetail />
                    </Layout>
                  </ProtectedRoute>
                } 
              />
              <Route 
                path="/transactions" 
                element={
                  <ProtectedRoute>
                    <Layout>
                      <Transactions />
                    </Layout>
                  </ProtectedRoute>
                } 
              />
              <Route 
                path="/transactions/:transactionId" 
                element={
                  <ProtectedRoute>
                    <Layout>
                      <TransactionDetail />
                    </Layout>
                  </ProtectedRoute>
                } 
              />
            </Routes>
          </div>
        </Router>
      </AuthProvider>
    </ThemeProvider>
  );
}

export default App;
