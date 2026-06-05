// Bearer-token handling for non-loopback ("token-protected") binds.
//
// Loopback binds need no token. When bound to a LAN host the server generates a token and
// prints a URL like http://host:port/?token=XXX. Opening that URL seeds the token into
// localStorage and strips it from the address bar. For browser-managed transports that
// cannot set an Authorization header (EventSource/SSE, <audio> src) the token is appended
// as a ?token= query parameter instead -- the backend's require_auth accepts both.

const TOKEN_KEY = "masr_token";

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null; // localStorage blocked (private mode); treat as no token
  }
}

export function setToken(token: string): void {
  try {
    localStorage.setItem(TOKEN_KEY, token);
  } catch {
    // localStorage unavailable; auth simply won't persist this session
  }
}

export function clearToken(): void {
  try {
    localStorage.removeItem(TOKEN_KEY);
  } catch {
    // ignore
  }
}

/** Pull ?token= out of the current URL into storage, then strip it from the address bar. */
export function seedTokenFromUrl(): void {
  const params = new URLSearchParams(window.location.search);
  const token = params.get("token");
  if (!token) return;
  setToken(token);
  params.delete("token");
  const query = params.toString();
  const url =
    window.location.pathname + (query ? `?${query}` : "") + window.location.hash;
  window.history.replaceState(null, "", url);
}

/** Append the stored token as a query param for non-fetch transports (SSE, <audio>). */
export function withToken(url: string): string {
  const token = getToken();
  if (!token) return url;
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}token=${encodeURIComponent(token)}`;
}
