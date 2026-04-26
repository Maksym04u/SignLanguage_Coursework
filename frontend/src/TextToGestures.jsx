import { useEffect, useRef, useState } from "react";

import { api } from "./api";
import { drawGestureFrame } from "./keypoints";

const CANVAS_SIZE = 160;

function GestureCard({ frame }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = CANVAS_SIZE;
    canvas.height = CANVAS_SIZE;
    if (frame.type === "missing" || frame.type === "space") {
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = frame.type === "missing" ? "#1f2937" : "#0f172a";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      return;
    }
    drawGestureFrame(canvas, frame.lh, frame.rh);
  }, [frame]);

  if (frame.type === "space") {
    return (
      <div className="gestureCard gestureCardSpace" aria-label="space">
        <div className="gestureSpaceBar" />
        <span className="gestureCardLabel gestureCardLabelMuted">space</span>
      </div>
    );
  }

  if (frame.type === "missing") {
    return (
      <div className="gestureCard gestureCardMissing" title="No gesture available">
        <canvas ref={canvasRef} className="gestureCanvas" />
        <span className="gestureCardBadge gestureCardBadgeMissing">missing</span>
        <span className="gestureCardLabel">{frame.label}</span>
      </div>
    );
  }

  const badgeClass =
    frame.type === "word" ? "gestureCardBadgeWord" : "gestureCardBadgeLetter";

  return (
    <div className="gestureCard">
      <canvas ref={canvasRef} className="gestureCanvas" />
      <span className={`gestureCardBadge ${badgeClass}`}>{frame.type}</span>
      <span className="gestureCardLabel">{frame.label}</span>
    </div>
  );
}

export function TextToGestures({ token, language }) {
  const [text, setText] = useState("");
  const [frames, setFrames] = useState([]);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const onTranslate = async () => {
    setError("");
    setSummary(null);
    if (!text.trim()) {
      setFrames([]);
      return;
    }
    setLoading(true);
    try {
      const data = await api(
        "/translate/text-to-sign",
        "POST",
        { source_language: language, text },
        token
      );
      setFrames(Array.isArray(data.frames) ? data.frames : []);
      setSummary(data.summary || null);
    } catch (e) {
      setError(String(e.message || e));
      setFrames([]);
    } finally {
      setLoading(false);
    }
  };

  const onClear = () => {
    setText("");
    setFrames([]);
    setSummary(null);
    setError("");
  };

  return (
    <section className="card">
      <h2>Text to Gestures</h2>
      <p className="cardSubtitle">
        Type a phrase and we will render the matching signs from your dataset.
        Whole-word gestures are used when available; otherwise the word is
        spelled out letter by letter.
      </p>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="e.g. hello name max"
        rows={3}
        className="textToSignInput"
      />

      <div className="row">
        <button type="button" onClick={onTranslate} disabled={loading}>
          {loading ? "Translating..." : "Translate"}
        </button>
        <button
          type="button"
          className="secondaryBtn"
          onClick={onClear}
          disabled={loading}
        >
          Clear
        </button>
      </div>

      {error ? (
        <div className="inlineMessage inlineError" role="alert">
          {error}
        </div>
      ) : null}

      {summary ? (
        <div className="textToSignSummary">
          <span>
            {summary.words_matched} word{summary.words_matched === 1 ? "" : "s"}
            , {summary.letters_matched} letter
            {summary.letters_matched === 1 ? "" : "s"}
          </span>
          {summary.letters_missing && summary.letters_missing.length > 0 ? (
            <span className="textToSignMissing">
              missing: {summary.letters_missing.join(", ")}
            </span>
          ) : null}
        </div>
      ) : null}

      {frames.length > 0 ? (
        <div className="gestureStrip">
          {frames.map((frame, idx) => (
            <GestureCard key={`${frame.type}-${idx}`} frame={frame} />
          ))}
        </div>
      ) : null}
    </section>
  );
}
