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

// ---------------------------------------------------------------------------
// Token expiry detection & refresh
// ---------------------------------------------------------------------------

/**
 * Check whether the current access token is expired or will expire
 * within `bufferSeconds` from now.
 *
 * JWT structure: header.payload.signature (base64url-encoded, dot-separated).
 * The payload contains an `exp` field (Unix timestamp in seconds).
 *
 * @param bufferSeconds - Treat the token as expired if it expires within
 *   this many seconds from now. Default 60.
 * @returns `true` if no token, token is malformed, or token expires within buffer.
 */
export function isTokenExpiringSoon(bufferSeconds = 60): boolean {
  try {
    const token = getAccessToken();
    if (!token) {
      return true;
    }

    const parts = token.split(".");
    if (parts.length !== 3) {
      return true;
    }

    // Base64url → standard base64
    const base64 = parts[1].replace(/-/g, "+").replace(/_/g, "/");
    const payload = JSON.parse(atob(base64)) as { exp?: number };

    if (typeof payload.exp !== "number") {
      return true;
    }

    return payload.exp < Date.now() / 1000 + bufferSeconds;
  } catch {
    return true;
  }
}

let _refreshPromise: Promise<string | null> | null = null;

async function _doRefreshToken(): Promise<string | null> {
  const refreshToken = getRefreshToken();
  if (!refreshToken) {
    return null;
  }

  try {
    const response = await fetch(apiUrl("/api/v1/auth/refresh"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });

    if (!response.ok) {
      return null;
    }

    const session = (await response.json()) as AuthSession;
    setAuthSession(session);
    return session.access_token;
  } catch {
    return null;
  }
}

/**
 * Refresh the access token using the stored refresh token.
 *
 * Concurrent calls are deduplicated — only one refresh request is in-flight
 * at a time, and all callers share the same result. This prevents the
 * one-time-use refresh token from being consumed by a race condition.
 *
 * @returns The new access token, or null if refresh failed.
 */
export function refreshAccessToken(): Promise<string | null> {
  if (!_refreshPromise) {
    _refreshPromise = _doRefreshToken().finally(() => {
      _refreshPromise = null;
    });
  }
  return _refreshPromise;
}

