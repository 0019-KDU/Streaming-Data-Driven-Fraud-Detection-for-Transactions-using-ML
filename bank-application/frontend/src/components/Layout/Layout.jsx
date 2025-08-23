import React, { useState } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Box,
  IconButton,
  Avatar,
  Menu,
  MenuItem,
  Badge,
  useTheme,
  useMediaQuery,
  Divider,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  People as PeopleIcon,
  Receipt as ReceiptIcon,
  AccountBalance,
  Menu as MenuIcon,
  Notifications,
  Settings,
  ExitToApp,
  ChevronLeft,
} from '@mui/icons-material';
import { useAuth } from '../../context/AuthContext';
import { useNavigate, useLocation } from 'react-router-dom';

const drawerWidth = 280;

const Layout = ({ children }) => {
  const { user, logout } = useAuth();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('lg'));
  const navigate = useNavigate();
  const location = useLocation();
  
  const [mobileOpen, setMobileOpen] = useState(false);
  const [anchorEl, setAnchorEl] = useState(null);

  const menuItems = [
    { text: 'Dashboard', icon: <DashboardIcon />, path: '/dashboard' },
    { text: 'Customers', icon: <PeopleIcon />, path: '/customers' },
    { text: 'Transactions', icon: <ReceiptIcon />, path: '/transactions' },
  ];

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleProfileMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleProfileMenuClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = () => {
    handleProfileMenuClose();
    logout();
  };

  const drawer = (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ p: 3, display: 'flex', alignItems: 'center', gap: 2 }}>
        <AccountBalance sx={{ fontSize: 32, color: 'primary.main' }} />
        <Typography variant="h6" sx={{ fontWeight: 700, color: 'primary.main' }}>
          BankFlow
        </Typography>
      </Box>
      
      <Divider />
      
      <List sx={{ flex: 1, pt: 2 }}>
        {menuItems.map((item) => (
          <ListItem 
            button 
            key={item.text}
            onClick={() => {
              navigate(item.path);
              if (isMobile) setMobileOpen(false);
            }}
            sx={{ 
              mb: 1, 
              mx: 2, 
              borderRadius: 2,
              backgroundColor: location.pathname === item.path ? 'primary.main' : 'transparent',
              color: location.pathname === item.path ? 'white' : 'text.primary',
              '&:hover': { 
                backgroundColor: location.pathname === item.path ? 'primary.dark' : 'action.hover',
              }
            }}
          >
            <ListItemIcon sx={{ 
              color: location.pathname === item.path ? 'white' : 'text.secondary',
              minWidth: 40 
            }}>
              {item.icon}
            </ListItemIcon>
            <ListItemText 
              primary={item.text} 
              primaryTypographyProps={{ 
                fontWeight: location.pathname === item.path ? 600 : 400 
              }}
            />
          </ListItem>
        ))}
      </List>
    </Box>
  );

  return (
    <Box sx={{ display: 'flex' }}>
      <AppBar 
        position="fixed" 
        sx={{ 
          zIndex: theme.zIndex.drawer + 1,
          backgroundColor: 'white',
          color: 'text.primary',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        }}
      >
        <Toolbar>
          {isMobile && (
            <IconButton
              color="inherit"
              aria-label="open drawer"
              edge="start"
              onClick={handleDrawerToggle}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>
          )}
          
          <Typography variant="h6" sx={{ flexGrow: 1, fontWeight: 600 }}>
            Bank Management System
          </Typography>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <IconButton color="inherit">
              <Badge badgeContent={3} color="error">
                <Notifications />
              </Badge>
            </IconButton>
            
            <IconButton onClick={handleProfileMenuOpen}>
              <Avatar sx={{ width: 32, height: 32, bgcolor: 'primary.main' }}>
                {user?.full_name?.[0] || 'A'}
              </Avatar>
            </IconButton>
          </Box>
        </Toolbar>
      </AppBar>

      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleProfileMenuClose}
        transformOrigin={{ horizontal: 'right', vertical: 'top' }}
        anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
      >
        <MenuItem onClick={handleProfileMenuClose}>
          <ListItemIcon><Settings fontSize="small" /></ListItemIcon>
          Settings
        </MenuItem>
        <MenuItem onClick={handleLogout}>
          <ListItemIcon><ExitToApp fontSize="small" /></ListItemIcon>
          Logout
        </MenuItem>
      </Menu>
      
      <Drawer
        variant={isMobile ? 'temporary' : 'permanent'}
        open={isMobile ? mobileOpen : true}
        onClose={handleDrawerToggle}
        ModalProps={{ keepMounted: true }}
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: drawerWidth,
            boxSizing: 'border-box',
            backgroundColor: '#fafafa',
            borderRight: '1px solid #e0e0e0',
          },
        }}
      >
        {drawer}
      </Drawer>
      
      <Box 
        component="main" 
        sx={{ 
          flexGrow: 1, 
          p: 0,
          mt: '64px',
          ml: isMobile ? 0 : `${drawerWidth}px`,
          minHeight: 'calc(100vh - 64px)',
          backgroundColor: 'background.default',
        }}
      >
        {children}
      </Box>
    </Box>
  );
};

export default Layout;
