import api from './api';

export const authAPI = {
  login: async (email, password) => {
    const response = await api.post('/auth/login', {
      email,
      password
    });
    return response.data;
  },

  getCurrentUser: async () => {
    const response = await api.get('/auth/me');
    return response.data;
  },

  logout: async () => {
    const response = await api.post('/auth/logout');
    return response.data;
  },

  createAdmin: async (adminData) => {
    const response = await api.post('/auth/create-admin', adminData);
    return response.data;
  }
};
