export const API_BASE = "http://127.0.0.1:8000";
export const LS_TOKEN = "sign_translator_access_token";
export const LS_EMAIL = "sign_translator_user_email";

export function persistSession(accessToken, email) {
  localStorage.setItem(LS_TOKEN, accessToken);
  if (email) {
    localStorage.setItem(LS_EMAIL, email);
  }
}

export function clearStoredSession() {
  localStorage.removeItem(LS_TOKEN);
  localStorage.removeItem(LS_EMAIL);
}

export function parseApiErrorMessage(text) {
  try {
    const j = JSON.parse(text);
    if (typeof j.detail === "string") {
      return j.detail;
    }
    if (Array.isArray(j.detail)) {
      return j.detail
        .map((d) => (typeof d === "object" && d.msg ? d.msg : JSON.stringify(d)))
        .join(" ");
    }
    if (j.detail != null) {
      return String(j.detail);
    }
  } catch {
    /* not JSON */
  }
  return text || "Request failed";
}

export async function api(path, method, body, token) {
  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {})
    },
    body: body ? JSON.stringify(body) : undefined
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(parseApiErrorMessage(text));
  }
  return res.json();
}
