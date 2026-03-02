"use client";

import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

import { apiUrl, API_BASE_URL } from "@/lib/api";
import {
  AuthSession,
  AuthUser,
  clearAuthSession,
  getAccessToken,
  getRefreshToken,
  getStoredUser,
  setAuthSession,
  tryServerLogout,
} from "@/lib/auth";

interface AuthContextValue {
  user: AuthUser | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

function getErrorMessage(defaultMessage: string, payload: unknown): string {
  if (payload && typeof payload === "object") {
    const detail = (payload as { detail?: unknown }).detail;
    if (typeof detail === "string" && detail.trim()) {
      return detail;
    }
    const message = (payload as { message?: unknown }).message;
    if (typeof message === "string" && message.trim()) {
      return message;
    }
  }
  return defaultMessage;
}

function isApiRequest(url: string): boolean {
  const base = API_BASE_URL.endsWith("/") ? API_BASE_URL.slice(0, -1) : API_BASE_URL;
  return url.startsWith(base);
}

async function parseJsonSafe(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [accessToken, setAccessToken] = useState<string | null>(null);

  const clearSessionRef = React.useRef<() => void>(() => {});
  const applySessionRef = React.useRef<(s: AuthSession) => void>(() => {});

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    const originalFetch = window.fetch.bind(window);

    const AUTH_ENDPOINTS = ["/auth/login", "/auth/register", "/auth/refresh", "/auth/logout"];

    function isAuthEndpoint(url: string): boolean {
      return AUTH_ENDPOINTS.some((ep) => url.includes(ep));
    }

    let refreshPromise: Promise<string | null> | null = null;

    async function tryRefreshToken(): Promise<string | null> {
      const refreshToken = getRefreshToken();
      if (!refreshToken) {
        return null;
      }

      try {
        const response = await originalFetch(apiUrl("/api/v1/auth/refresh"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ refresh_token: refreshToken }),
        });

        if (!response.ok) {
          return null;
        }

        const session = (await response.json()) as AuthSession;
        applySessionRef.current(session);
        return session.access_token;
      } catch {
        return null;
      }
    }

    function refreshOnce(): Promise<string | null> {
      if (!refreshPromise) {
        refreshPromise = tryRefreshToken().finally(() => {
          refreshPromise = null;
        });
      }
      return refreshPromise;
    }

    function doFetch(
      input: RequestInfo | URL,
      init: RequestInit | undefined,
      token: string,
    ): Promise<Response> {
      if (input instanceof Request) {
        const request = new Request(input, init);
        if (!request.headers.has("Authorization")) {
          request.headers.set("Authorization", `Bearer ${token}`);
        }
        return originalFetch(request);
      }

      const headers = new Headers(init?.headers);
      if (!headers.has("Authorization")) {
        headers.set("Authorization", `Bearer ${token}`);
      }

      return originalFetch(input, { ...init, headers });
    }

    window.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const token = getAccessToken();
      if (!token) {
        return originalFetch(input, init);
      }

      const url =
        typeof input === "string"
          ? input
          : input instanceof URL
            ? input.toString()
            : input.url;

      if (!isApiRequest(url)) {
        return originalFetch(input, init);
      }

      if (isAuthEndpoint(url)) {
        return originalFetch(input, init);
      }

      let response: Response;
      try {
        response = await doFetch(input, init, token);
      } catch (err) {
        // Network-level failures should not force logout; keep session and let caller handle error.
        throw err;
      }

      if (response.status !== 401) {
        return response;
      }

      const newToken = await refreshOnce();
      if (newToken) {
        return doFetch(input, init, newToken);
      }

      clearSessionRef.current();
      return response;
    };

    return () => {
      window.fetch = originalFetch;
    };
  }, []);

  const applySession = useCallback((session: AuthSession) => {
    setAuthSession(session);
    setAccessToken(session.access_token);
    setUser(session.user);
  }, []);

  const clearSession = useCallback(() => {
    clearAuthSession();
    setAccessToken(null);
    setUser(null);
  }, []);

  useEffect(() => {
    clearSessionRef.current = clearSession;
    applySessionRef.current = applySession;
  }, [clearSession, applySession]);

  const refresh = useCallback(async (): Promise<AuthSession | null> => {
    const refreshToken = getRefreshToken();
    if (!refreshToken) {
      return null;
    }

    const response = await fetch(apiUrl("/api/v1/auth/refresh"), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });

    if (!response.ok) {
      return null;
    }

    const payload = (await response.json()) as AuthSession;
    applySession(payload);
    return payload;
  }, [applySession]);

  const verify = useCallback(async (): Promise<boolean> => {
    const token = getAccessToken();
    if (!token) {
      clearSession();
      return false;
    }

    const response = await fetch(apiUrl("/api/v1/auth/me"), {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });

    if (response.ok) {
      const payload = (await response.json()) as { user: AuthUser };
      setAccessToken(token);
      setUser(payload.user);
      return true;
    }

    const refreshed = await refresh();
    if (!refreshed) {
      clearSession();
      return false;
    }

    return true;
  }, [clearSession, refresh]);

  useEffect(() => {
    let mounted = true;

    const initialize = async () => {
      const token = getAccessToken();
      const cachedUser = getStoredUser();

      if (!token) {
        clearSession();
        if (mounted) {
          setIsLoading(false);
        }
        return;
      }

      if (cachedUser) {
        setUser(cachedUser);
      }
      setAccessToken(token);

      try {
        await verify();
      } finally {
        if (mounted) {
          setIsLoading(false);
        }
      }
    };

    initialize();

    return () => {
      mounted = false;
    };
  }, [clearSession, verify]);

  const login = useCallback(
    async (email: string, password: string) => {
      const response = await fetch(apiUrl("/api/v1/auth/login"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email, password }),
      });

      const payload = await parseJsonSafe(response);
      if (!response.ok) {
        throw new Error(getErrorMessage("Login failed", payload));
      }

      applySession(payload as AuthSession);
    },
    [applySession],
  );

  const register = useCallback(
    async (email: string, password: string) => {
      const response = await fetch(apiUrl("/api/v1/auth/register"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email, password }),
      });

      const payload = await parseJsonSafe(response);
      if (!response.ok) {
        throw new Error(getErrorMessage("Register failed", payload));
      }

      await login(email, password);
    },
    [login],
  );

  const logout = useCallback(async () => {
    await tryServerLogout();
    clearSession();
  }, [clearSession]);

  const value = useMemo<AuthContextValue>(
    () => ({
      user,
      isAuthenticated: Boolean(user && accessToken),
      isLoading,
      login,
      register,
      logout,
    }),
    [accessToken, isLoading, login, logout, register, user],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return context;
}

