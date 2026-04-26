import { useEffect, useMemo, useState } from "react";

import {
  api,
  clearStoredSession,
  LS_EMAIL,
  LS_TOKEN,
  persistSession
} from "./api";
import { LiveTranslator } from "./LiveTranslator";
import { TextToGestures } from "./TextToGestures";

export function App() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [preferredLanguage, setPreferredLanguage] = useState("en");
  const [token, setToken] = useState(() => localStorage.getItem(LS_TOKEN) ?? "");
  const [userEmail, setUserEmail] = useState(() => localStorage.getItem(LS_EMAIL) ?? "");
  const [authMode, setAuthMode] = useState("login");
  const [language, setLanguage] = useState("en");
  const [history, setHistory] = useState([]);
  const [authError, setAuthError] = useState("");
  const [historyError, setHistoryError] = useState("");

  const isAuthed = useMemo(() => Boolean(token), [token]);

  useEffect(() => {
    if (!isAuthed || !token) {
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const me = await api("/auth/me", "GET", null, token);
        if (!cancelled && me.email) {
          setUserEmail(me.email);
          localStorage.setItem(LS_EMAIL, me.email);
        }
        if (!cancelled && me.preferred_language) {
          setLanguage(me.preferred_language);
        }
      } catch {
        if (!cancelled) {
          clearStoredSession();
          setToken("");
          setUserEmail("");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [isAuthed, token]);

  const switchAuthMode = (mode) => {
    setAuthMode(mode);
    setAuthError("");
  };

  const onSignup = async () => {
    setAuthError("");
    try {
      const data = await api("/auth/signup", "POST", {
        email,
        password,
        preferred_language: preferredLanguage
      });
      setToken(data.access_token);
      if (data.email) {
        setUserEmail(data.email);
      }
      setLanguage(preferredLanguage);
      persistSession(data.access_token, data.email);
    } catch (e) {
      setAuthError(String(e.message || e));
    }
  };

  const onLogin = async () => {
    setAuthError("");
    try {
      const data = await api("/auth/login", "POST", { email, password });
      setToken(data.access_token);
      if (data.email) {
        setUserEmail(data.email);
      }
      persistSession(data.access_token, data.email);
    } catch (e) {
      setAuthError(String(e.message || e));
    }
  };

  const onLoadHistory = async () => {
    setHistoryError("");
    try {
      const data = await api("/translate/history", "GET", null, token);
      setHistory(data);
    } catch (e) {
      setHistoryError(String(e.message || e));
    }
  };

  const onLogout = () => {
    clearStoredSession();
    setToken("");
    setUserEmail("");
    setHistory([]);
    setAuthError("");
    setHistoryError("");
  };

  return (
    <main className="container">
      <header className="topBar">
        <div className="topBarLeft">
          {isAuthed && userEmail ? <span className="userEmail">{userEmail}</span> : null}
          <div className="topBarTitles">
            <h1>Multilingual Sign Translator</h1>
            <p>Translate sign streams into readable text.</p>
          </div>
        </div>
        {isAuthed ? (
          <button type="button" className="logoutBtn" onClick={onLogout}>
            Logout
          </button>
        ) : null}
      </header>

      {isAuthed ? (
        <>
          <section className="card">
            <h2>Translation Language</h2>
            <select value={language} onChange={(e) => setLanguage(e.target.value)}>
              <option value="en">English</option>
              <option value="uk">Ukrainian</option>
            </select>
          </section>

          <LiveTranslator
            token={token}
            language={language}
            onSaved={() => {
              onLoadHistory();
            }}
          />

          <TextToGestures token={token} language={language} />

          <section className="card">
            <div className="row spaceBetween">
              <h2>History</h2>
              <button type="button" onClick={onLoadHistory}>
                Refresh
              </button>
            </div>
            {historyError ? (
              <div className="inlineMessage inlineError" role="alert">
                {historyError}
              </div>
            ) : null}
            {history.length === 0 ? (
              <p>No translations saved yet.</p>
            ) : (
              history.map((item) => (
                <div key={item.id} className="historyItem">
                  <span>{item.source_language}</span>
                  <span>{item.raw_text}</span>
                  <span>{item.corrected_text || "-"}</span>
                  <span>{item.created_at}</span>
                </div>
              ))
            )}
          </section>
        </>
      ) : null}

      {!isAuthed ? (
        <div className="authOverlay">
          <div className="authModal">
            <h2>{authMode === "login" ? "Login" : "Sign Up"}</h2>
            <p>
              {authMode === "login"
                ? "Please log in to continue."
                : "Create an account to continue."}
            </p>
            {authError ? (
              <div className="inlineMessage inlineError" role="alert">
                {authError}
              </div>
            ) : null}
            <div className="fieldStack">
              <input
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="Email"
                autoComplete="email"
              />
              <input
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                type="password"
                placeholder="Password"
                autoComplete={authMode === "login" ? "current-password" : "new-password"}
              />
              {authMode === "signup" ? (
                <select
                  value={preferredLanguage}
                  onChange={(e) => setPreferredLanguage(e.target.value)}
                >
                  <option value="en">Preferred language: English</option>
                  <option value="uk">Preferred language: Ukrainian</option>
                </select>
              ) : null}
            </div>
            <div className="row modalActions">
              {authMode === "login" ? (
                <button type="button" onClick={onLogin}>
                  Login
                </button>
              ) : (
                <button type="button" onClick={onSignup}>
                  Sign Up
                </button>
              )}
              <button
                type="button"
                className="secondaryBtn"
                onClick={() => switchAuthMode(authMode === "login" ? "signup" : "login")}
              >
                {authMode === "login" ? "Need account?" : "Have account?"}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </main>
  );
}
