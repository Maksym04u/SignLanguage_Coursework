import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api } from "./api";
import {
  computeSequenceBounds,
  drawGestureFrame,
  isRawPlaybackFrame,
  prepareRawPlaybackSequence,
} from "./keypoints";

// Animation knobs (tuned per user feedback)
// 12 fps recording rate -> 20 frames @ 83 ms = ~1.66 s of motion.
const FRAME_MS = 83;
// Each visible gesture freezes on its first pose for COOLDOWN_MS so the user
// has time to spot where it begins before the motion starts. This fixes the
// "moving gestures disappear when the previous one finishes" feeling: with
// per-sequence bounding boxes, the new gesture's first frame lands wherever
// its own motion begins (e.g. ц starts near the top), which is usually far
// from where the previous gesture ended.
const COOLDOWN_MS = 500;
const GESTURE_HOLD_MS = 180; // brief hold on the final pose before advancing
// Silent frames (whitespace / punctuation) flash through the sentence builder
// without disturbing the stage. A tiny delay keeps the "typing" feel.
const SILENT_MS = 70;
const MISSING_MS = 1100;

// Rendering: 4:3 internal resolution matching the live-translation stage.
const STAGE_WIDTH = 640;
const STAGE_HEIGHT = 480;

const SILENT_TYPES = new Set(["silent", "space"]);

function gestureDurationMs(frame) {
  if (!frame) return 0;
  if (SILENT_TYPES.has(frame.type)) return SILENT_MS;
  if (frame.type === "missing") return MISSING_MS;
  const seqLen =
    Array.isArray(frame.sequence) && frame.sequence.length > 0
      ? frame.sequence.length
      : 1;
  return COOLDOWN_MS + seqLen * FRAME_MS + GESTURE_HOLD_MS;
}

function PlaybackStage({ frame, displaySequence, bounds, playing, done }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    if (canvas.width !== STAGE_WIDTH) canvas.width = STAGE_WIDTH;
    if (canvas.height !== STAGE_HEIGHT) canvas.height = STAGE_HEIGHT;
    const ctx = canvas.getContext("2d");

    const paintBackground = (color = "#0f172a") => {
      ctx.fillStyle = color;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    };

    const drawOptions = {
      bounds,
      playbackFormat: frame?.playback_format ?? null,
    };

    if (!frame) {
      paintBackground();
      if (done) {
        ctx.fillStyle = "rgba(148, 163, 184, 0.85)";
        ctx.font = "600 28px Arial, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(
          "Press Replay to play it again",
          canvas.width / 2,
          canvas.height / 2
        );
      }
      return undefined;
    }

    if (frame.type === "missing") {
      paintBackground("#3f1d1d");
      ctx.fillStyle = "rgba(248, 113, 113, 0.95)";
      ctx.font = "600 36px Arial, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(
        `no gesture for "${frame.label}"`,
        canvas.width / 2,
        canvas.height / 2
      );
      return undefined;
    }

    const seq = Array.isArray(displaySequence) ? displaySequence : null;

    if (!seq || seq.length === 0) {
      drawGestureFrame(canvas, frame.lh, frame.rh, drawOptions);
      return undefined;
    }

    if (!playing) {
      const last = seq[seq.length - 1];
      drawGestureFrame(canvas, last.lh, last.rh, drawOptions);
      return undefined;
    }

    let raf = 0;
    let cancelled = false;
    const start = performance.now();
    drawGestureFrame(canvas, seq[0].lh, seq[0].rh, drawOptions);

    const loop = (now) => {
      if (cancelled) return;
      const elapsed = now - start;
      if (elapsed < COOLDOWN_MS) {
        raf = requestAnimationFrame(loop);
        return;
      }
      const animElapsed = elapsed - COOLDOWN_MS;
      const idx = Math.min(seq.length - 1, Math.floor(animElapsed / FRAME_MS));
      const pose = seq[idx];
      drawGestureFrame(canvas, pose.lh, pose.rh, drawOptions);
      if (idx >= seq.length - 1) return;
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);

    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
    };
  }, [frame, displaySequence, bounds, playing, done]);

  return <canvas ref={canvasRef} className="gestureStageCanvas" />;
}

export function TextToGestures({ token, language }) {
  const [text, setText] = useState("");
  const [frames, setFrames] = useState([]);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // -1 means "not playing". When >=0 it's the index of the frame currently on
  // stage; its label will be appended to the built sentence when its timer
  // fires, then we advance to the next.
  const [activeIdx, setActiveIdx] = useState(-1);
  const [revealedCount, setRevealedCount] = useState(0);
  const timerRef = useRef(0);

  const clearTimer = useCallback(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = 0;
    }
  }, []);

  useEffect(() => {
    clearTimer();
    setActiveIdx(-1);
    setRevealedCount(0);
  }, [frames, clearTimer]);

  useEffect(() => {
    if (frames.length > 0) setActiveIdx(0);
  }, [frames]);

  useEffect(() => {
    clearTimer();
    if (activeIdx < 0 || activeIdx >= frames.length) return undefined;
    const current = frames[activeIdx];
    const duration = gestureDurationMs(current);

    timerRef.current = window.setTimeout(() => {
      setRevealedCount((c) => Math.max(c, activeIdx + 1));
      if (activeIdx + 1 < frames.length) {
        setActiveIdx(activeIdx + 1);
      } else {
        setActiveIdx(-1);
      }
    }, duration);

    return clearTimer;
  }, [activeIdx, frames, clearTimer]);

  useEffect(() => () => clearTimer(), [clearTimer]);

  const isPlaybackRunning = activeIdx >= 0;
  const done =
    !isPlaybackRunning && frames.length > 0 && revealedCount >= frames.length;

  // The "stage" shows only word/letter/missing gestures -- silent frames
  // (spaces, punctuation) flash through the sentence builder without
  // interrupting the visual rhythm. We anchor `stageFrame` on the most
  // recent visible frame so silent transitions look like a held pose.
  const stageFrame = useMemo(() => {
    if (!frames.length) return null;
    const startIdx = isPlaybackRunning
      ? activeIdx
      : Math.min(revealedCount, frames.length) - 1;
    for (let i = startIdx; i >= 0; i--) {
      const f = frames[i];
      if (f && !SILENT_TYPES.has(f.type)) return f;
    }
    return null;
  }, [frames, activeIdx, revealedCount, isPlaybackRunning]);

  // The stage should animate only when the controller is sitting on the
  // stageFrame; while silent frames pass by, freeze on the last pose.
  const stageIsAnimating =
    isPlaybackRunning && frames[activeIdx] === stageFrame;

  const displaySequence = useMemo(() => {
    if (!stageFrame?.sequence?.length) return null;
    const first = stageFrame.sequence[0];
    if (
      isRawPlaybackFrame(
        first?.lh,
        first?.rh,
        stageFrame.playback_format ?? null
      )
    ) {
      return prepareRawPlaybackSequence(stageFrame.sequence);
    }
    return stageFrame.sequence;
  }, [stageFrame]);

  // One shared bounding box per clip (after raw hand-size standardization)
  // so motion trajectories are preserved frame-to-frame.
  const stageBounds = useMemo(() => {
    if (displaySequence?.length) return computeSequenceBounds(displaySequence);
    if (!stageFrame || !Array.isArray(stageFrame.sequence)) return null;
    return computeSequenceBounds(stageFrame.sequence);
  }, [displaySequence, stageFrame]);

  const builtSentence = useMemo(() => {
    if (!frames.length) return "";
    return frames
      .slice(0, revealedCount)
      .map((f) => f.label || "")
      .join("");
  }, [frames, revealedCount]);

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

  const onReplay = () => {
    if (!frames.length) return;
    clearTimer();
    setRevealedCount(0);
    setActiveIdx(0);
  };

  const onClear = () => {
    clearTimer();
    setText("");
    setFrames([]);
    setSummary(null);
    setError("");
    setActiveIdx(-1);
    setRevealedCount(0);
  };

  const hasFrames = frames.length > 0;

  // Progress counts only stageable (visible) frames so the user sees a clean
  // "3 of 5 gestures" rather than something inflated by punctuation.
  const visibleFrames = useMemo(
    () => frames.filter((f) => !SILENT_TYPES.has(f.type)),
    [frames]
  );

  const visiblePlayed = useMemo(
    () =>
      frames
        .slice(0, Math.min(revealedCount, frames.length))
        .filter((f) => !SILENT_TYPES.has(f.type)).length,
    [frames, revealedCount]
  );

  const progressLabel = visibleFrames.length
    ? `${Math.min(
        isPlaybackRunning ? visiblePlayed + (stageIsAnimating ? 1 : 0) : visiblePlayed,
        visibleFrames.length
      )} / ${visibleFrames.length}`
    : "";

  return (
    <section className="card">
      <h2>Text to Gestures</h2>
      <p className="cardSubtitle">
        Type a phrase and we will play the matching signs from your dataset.
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
          onClick={onReplay}
          disabled={loading || !hasFrames}
        >
          Replay
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

      {hasFrames ? (
        <>
          <div className="gesturePlaybackBar">
            <div className="gesturePlaybackLabel">
              {isPlaybackRunning
                ? "Building sentence..."
                : done
                ? "Complete"
                : "Ready"}
            </div>
            <p className="gesturePlaybackText">
              {builtSentence || (
                <span className="gesturePlaybackPlaceholder">&nbsp;</span>
              )}
              {isPlaybackRunning ? (
                <span className="gesturePlaybackCaret">|</span>
              ) : null}
            </p>
          </div>

          <div className="gestureStage">
            <PlaybackStage
              frame={stageFrame}
              displaySequence={displaySequence}
              bounds={stageBounds}
              playing={stageIsAnimating}
              done={done}
            />
            <div className="gestureStageOverlay">
              {stageFrame ? (
                <>
                  <span
                    className={`gestureStageBadge gestureStageBadge-${stageFrame.type}`}
                  >
                    {stageFrame.type}
                  </span>
                  <span className="gestureStageCurrentLabel">
                    {stageFrame.label}
                  </span>
                </>
              ) : (
                <span className="gestureStageBadge gestureStageBadge-ready">
                  ready
                </span>
              )}
              <span className="gestureStageProgress">{progressLabel}</span>
            </div>
          </div>
        </>
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
    </section>
  );
}
