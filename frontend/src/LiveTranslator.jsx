import { useCallback, useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

import { api } from "./api";
import { drawHandLandmarks, extractKeypoints } from "./keypoints";

const BUFFER_SIZE = 20;
const CONFIDENCE_THRESHOLD = 0.96;
const PREDICTION_INTERVAL_MS = 700;
const BUFFER_RESET_DELAY_MS = 750;
const BUFFER_TIMEOUT_MS = 1000;
const RECOGNITION_FLASH_MS = 1000;

const VISION_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.20/wasm";
const HAND_LANDMARKER_TASK =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

function isSingleLetter(text) {
  return typeof text === "string" && text.length === 1 && /[A-Za-z]/.test(text);
}

function isLettersOnlyToken(token) {
  return typeof token === "string" && /^[A-Za-z]+$/.test(token);
}

export function LiveTranslator({ token, language }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const handLandmarkerRef = useRef(null);
  const rafRef = useRef(0);

  const bufferRef = useRef([]);
  const lastPredictionTimeRef = useRef(0);
  const lastLandmarkTimeRef = useRef(0);
  const bufferResetTimeRef = useRef(0);
  const inFlightRef = useRef(false);

  const [status, setStatus] = useState("Idle. Click Start to begin.");
  const [running, setRunning] = useState(false);
  const [confidence, setConfidence] = useState(0);
  const [detected, setDetected] = useState("");
  const [bufferSize, setBufferSize] = useState(0);
  const [sentence, setSentence] = useState([]);
  const [recognizedFlash, setRecognizedFlash] = useState("");
  const [error, setError] = useState("");
  const [savedResult, setSavedResult] = useState(null);

  const drawIdleCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    if (canvas.width < 2) canvas.width = 640;
    if (canvas.height < 2) canvas.height = 480;
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }, []);

  const resetBuffer = useCallback(() => {
    bufferRef.current = [];
    setBufferSize(0);
  }, []);

  const appendSign = useCallback((sign) => {
    if (!sign) return;
    setSentence((prev) => {
      const next = [...prev];
      if (isSingleLetter(sign)) {
        const lower = sign.toLowerCase();
        const last = next[next.length - 1];
        if (last && isLettersOnlyToken(last)) {
          next[next.length - 1] = last + lower;
        } else {
          next.push(lower);
        }
      } else {
        next.push(sign);
      }
      if (next.length === 1 && next[0]) {
        next[0] = next[0].charAt(0).toUpperCase() + next[0].slice(1);
      }
      return next;
    });
  }, []);

  const ensureLandmarker = useCallback(async () => {
    if (handLandmarkerRef.current) return handLandmarkerRef.current;
    setStatus("Loading hand model...");
    const fileset = await FilesetResolver.forVisionTasks(VISION_WASM);
    const lm = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: HAND_LANDMARKER_TASK },
      runningMode: "VIDEO",
      numHands: 2
    });
    handLandmarkerRef.current = lm;
    return lm;
  }, []);

  const sendPrediction = useCallback(
    async (snapshot) => {
      try {
        const data = await api(
          "/translate/predict",
          "POST",
          { keypoints: snapshot, source_language: language },
          token
        );
        setConfidence(data.confidence);
        setDetected(data.display_text || "");
        if (data.confidence >= CONFIDENCE_THRESHOLD) {
          appendSign(data.display_text);
          resetBuffer();
          bufferResetTimeRef.current = performance.now() + BUFFER_RESET_DELAY_MS;
          setRecognizedFlash(data.display_text);
          window.setTimeout(() => setRecognizedFlash(""), RECOGNITION_FLASH_MS);
        }
      } catch (e) {
        setError(String(e.message || e));
      }
    },
    [appendSign, language, resetBuffer, token]
  );

  const tick = useCallback(() => {
    const lm = handLandmarkerRef.current;
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!lm || !video || !canvas) {
      rafRef.current = requestAnimationFrame(tick);
      return;
    }

    if (
      video.readyState >= 2 &&
      video.videoWidth > 0 &&
      video.videoHeight > 0
    ) {
      const ctx = canvas.getContext("2d");
      if (canvas.width !== video.videoWidth) canvas.width = video.videoWidth;
      if (canvas.height !== video.videoHeight) canvas.height = video.videoHeight;

      // Mirror horizontally so MediaPipe sees what the user expects, matching
      // the cv2.flip(image, 1) call done in main.py and during data collection.
      ctx.save();
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      ctx.restore();

      let result;
      try {
        result = lm.detectForVideo(canvas, performance.now());
      } catch (err) {
        setError(`Detection error: ${err.message || err}`);
        rafRef.current = requestAnimationFrame(tick);
        return;
      }

      let leftHand = null;
      let rightHand = null;
      const hands = result?.landmarks ?? [];
      const handed = result?.handednesses ?? [];
      for (let i = 0; i < hands.length; i++) {
        const label = handed[i]?.[0]?.categoryName ?? (i === 0 ? "Left" : "Right");
        if (label === "Left" && !leftHand) {
          leftHand = hands[i];
        } else if (label === "Right" && !rightHand) {
          rightHand = hands[i];
        } else if (!leftHand) {
          leftHand = hands[i];
        } else if (!rightHand) {
          rightHand = hands[i];
        }
      }

      drawHandLandmarks(ctx, hands);

      const now = performance.now();
      const handsDetected = Boolean(leftHand) || Boolean(rightHand);

      if (
        bufferRef.current.length > 0 &&
        now - lastLandmarkTimeRef.current > BUFFER_TIMEOUT_MS
      ) {
        resetBuffer();
      }
      if (bufferResetTimeRef.current > 0 && now >= bufferResetTimeRef.current) {
        resetBuffer();
        bufferResetTimeRef.current = 0;
      }

      if (handsDetected) {
        const keypoints = extractKeypoints({ leftHand, rightHand });
        lastLandmarkTimeRef.current = now;
        if (bufferResetTimeRef.current === 0) {
          bufferRef.current.push(keypoints);
          if (bufferRef.current.length > BUFFER_SIZE) {
            bufferRef.current.shift();
          }
          setBufferSize(bufferRef.current.length);

          if (
            bufferRef.current.length === BUFFER_SIZE &&
            now - lastPredictionTimeRef.current >= PREDICTION_INTERVAL_MS &&
            !inFlightRef.current
          ) {
            inFlightRef.current = true;
            lastPredictionTimeRef.current = now;
            const snapshot = bufferRef.current.slice();
            sendPrediction(snapshot).finally(() => {
              inFlightRef.current = false;
            });
          }
        }
      }
    }

    rafRef.current = requestAnimationFrame(tick);
  }, [resetBuffer, sendPrediction]);

  const stop = useCallback(() => {
    setRunning(false);
    cancelAnimationFrame(rafRef.current);
    rafRef.current = 0;
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    drawIdleCanvas();
    setStatus("Stopped.");
  }, [drawIdleCanvas]);

  const start = useCallback(async () => {
    setError("");
    try {
      await ensureLandmarker();
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false
      });
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) {
        throw new Error("Video element not ready");
      }
      video.srcObject = stream;
      await video.play();
      resetBuffer();
      lastPredictionTimeRef.current = 0;
      lastLandmarkTimeRef.current = performance.now();
      bufferResetTimeRef.current = 0;
      setRunning(true);
      setStatus("Translating live...");
      rafRef.current = requestAnimationFrame(tick);
    } catch (e) {
      setError(`Camera/model error: ${e.message || e}`);
      stop();
    }
  }, [ensureLandmarker, resetBuffer, stop, tick]);

  useEffect(() => {
    drawIdleCanvas();
    return () => {
      cancelAnimationFrame(rafRef.current);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
    };
  }, [drawIdleCanvas]);

  const onAddSpace = () => {
    setSentence((prev) => (prev.length ? [...prev, " "] : prev));
  };

  const onClear = () => {
    setSentence([]);
    setSavedResult(null);
    setRecognizedFlash("");
    setDetected("");
    setConfidence(0);
  };

  const onSave = async () => {
    if (sentence.length === 0) return;
    setError("");
    try {
      const data = await api(
        "/translate/sign-to-text",
        "POST",
        { source_language: language, tokens: sentence },
        token
      );
      setSavedResult(data);
    } catch (e) {
      setError(String(e.message || e));
    }
  };

  const sentenceText = sentence.join("");
  const confidencePct = Math.max(0, Math.min(100, confidence * 100));
  const isHighConfidence = confidence >= CONFIDENCE_THRESHOLD;

  return (
    <section className="card cardStack">
      <h2>Live Translation</h2>

      <div className="liveMainLayout">
        <div className="liveStreamWrap">
          <video ref={videoRef} className="liveVideoHidden" muted playsInline />
          <canvas ref={canvasRef} className="liveCanvas" />
          {recognizedFlash ? (
            <div className="recognizedFlash">{recognizedFlash}</div>
          ) : null}
        </div>

        <div className="liveSidePanel">
          <div className="liveStatusGrid">
            <div className="liveStatusItem">
              <span className="liveStatusLabel">Status</span>
              <span>{status}</span>
            </div>
            <div className="liveStatusItem">
              <span className="liveStatusLabel">Buffer</span>
              <span>
                {bufferSize}/{BUFFER_SIZE} frames
              </span>
            </div>
            <div className="liveStatusItem">
              <span className="liveStatusLabel">Detected</span>
              <span>{detected || "—"}</span>
            </div>
          </div>

          <div className="confidenceBar" aria-label="Confidence bar">
            <div
              className="confidenceFill"
              style={{
                width: `${confidencePct}%`,
                background: isHighConfidence ? "#10b981" : "#f59e0b"
              }}
            />
            <span className="confidenceText">
              Confidence: {(confidence * 100).toFixed(1)}%
            </span>
          </div>

          <div className="row">
            {!running ? (
              <button type="button" onClick={start}>
                Start
              </button>
            ) : (
              <button type="button" className="logoutBtn" onClick={stop}>
                Stop
              </button>
            )}
            <button
              type="button"
              className="secondaryBtn"
              onClick={onAddSpace}
              disabled={!sentence.length}
            >
              Add space
            </button>
            <button
              type="button"
              className="secondaryBtn"
              onClick={onClear}
              disabled={!sentence.length}
            >
              Clear
            </button>
            <button type="button" onClick={onSave} disabled={!sentence.length}>
              Save & Correct
            </button>
          </div>

          <div className="sentencePanel">
            <h3>Recognized Text</h3>
            <p className="sentenceText">{sentenceText || <em>Start signing to see text here.</em>}</p>
          </div>

          {savedResult ? (
            <div>
              <p>
                <b>Raw:</b> {savedResult.raw_text}
              </p>
              <p>
                <b>Corrected:</b> {savedResult.corrected_text || "(none)"}
              </p>
            </div>
          ) : null}

          {error ? (
            <div className="inlineMessage inlineError" role="alert">
              {error}
            </div>
          ) : null}
        </div>
      </div>
    </section>
  );
}
