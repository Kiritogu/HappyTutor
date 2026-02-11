import { apiUrl } from "@/lib/api";

export const ACCESS_TOKEN_STORAGE_KEY = "deeptutor-auth-access-token";
export const REFRESH_TOKEN_STORAGE_KEY = "deeptutor-auth-refresh-token";
export const USER_STORAGE_KEY = "deeptutor-auth-user";

export interface AuthUser {
  id: string;
  email: string;
  is_email_verified: boolean;
  status: string;
}

export interface AuthSession {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
  user: AuthUser;
}

function isBrowser(): boolean {
  return typeof window !== "undefined";
}

export function getAccessToken(): string | null {
  if (!isBrowser()) {
    return null;
  }
  return localStorage.getItem(ACCESS_TOKEN_STORAGE_KEY);
}

export function getRefreshToken(): string | null {
  if (!isBrowser()) {
    return null;
  }
  return localStorage.getItem(REFRESH_TOKEN_STORAGE_KEY);
}

export function getStoredUser(): AuthUser | null {
  if (!isBrowser()) {
    return null;
  }

  const raw = localStorage.getItem(USER_STORAGE_KEY);
  if (!raw) {
    return null;
  }

  try {
    return JSON.parse(raw) as AuthUser;
  } catch {
    return null;
  }
}

export function setAuthSession(session: AuthSession): void {
  if (!isBrowser()) {
    return;
  }

  localStorage.setItem(ACCESS_TOKEN_STORAGE_KEY, session.access_token);
  localStorage.setItem(REFRESH_TOKEN_STORAGE_KEY, session.refresh_token);
  localStorage.setItem(USER_STORAGE_KEY, JSON.stringify(session.user));
}

export function clearAuthSession(): void {
  if (!isBrowser()) {
    return;
  }

  localStorage.removeItem(ACCESS_TOKEN_STORAGE_KEY);
  localStorage.removeItem(REFRESH_TOKEN_STORAGE_KEY);
  localStorage.removeItem(USER_STORAGE_KEY);
}

export async function tryServerLogout(): Promise<void> {
  if (!isBrowser()) {
    return;
  }

  const refreshToken = getRefreshToken();
  if (!refreshToken) {
    return;
  }

  try {
    await fetch(apiUrl("/api/v1/auth/logout"), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });
  } catch {
    // Ignore network/logout errors on client-side cleanup.
  }
}

