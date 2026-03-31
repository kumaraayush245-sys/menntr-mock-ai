import { create } from 'zustand';
import { apiClient } from '@/lib/api/client';

export interface User {
  id: number;
  email: string;
  full_name: string | null;
  is_active: boolean;
  is_verified: boolean;
  created_at: string;
}

interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;

  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, fullName: string) => Promise<void>;
  logout: () => void;
  fetchUser: () => Promise<void>;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  isAuthenticated: false,
  isLoading: true,

  login: async (email, password) => {
    set({ isLoading: true });
    try {
      const data = await apiClient.post<{ access_token: string; token_type: string }>(
        '/api/v1/auth/login',
        { email, password }
      );
      localStorage.setItem('auth_token', data.access_token);

      const user = await apiClient.get<User>('/api/v1/auth/me');
      set({ user, isAuthenticated: true, isLoading: false });
    } catch (error) {
      set({ isLoading: false });
      throw error;
    }
  },

  register: async (email, password, fullName) => {
    set({ isLoading: true });
    try {
      await apiClient.post('/api/v1/auth/register', {
        email,
        password,
        full_name: fullName,
      });

      // Auto-login after registration
      const data = await apiClient.post<{ access_token: string; token_type: string }>(
        '/api/v1/auth/login',
        { email, password }
      );
      localStorage.setItem('auth_token', data.access_token);

      const user = await apiClient.get<User>('/api/v1/auth/me');
      set({ user, isAuthenticated: true, isLoading: false });
    } catch (error) {
      set({ isLoading: false });
      throw error;
    }
  },

  logout: () => {
    localStorage.removeItem('auth_token');
    set({ user: null, isAuthenticated: false, isLoading: false });
  },

  fetchUser: async () => {
    const token = typeof window !== 'undefined' ? localStorage.getItem('auth_token') : null;
    if (!token) {
      set({ user: null, isAuthenticated: false, isLoading: false });
      return;
    }
    try {
      const user = await apiClient.get<User>('/api/v1/auth/me');
      set({ user, isAuthenticated: true, isLoading: false });
    } catch {
      localStorage.removeItem('auth_token');
      set({ user: null, isAuthenticated: false, isLoading: false });
    }
  },
}));

// Initialise auth state on load (client-side only)
if (typeof window !== 'undefined') {
  useAuthStore.getState().fetchUser();
}
