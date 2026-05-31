// Mirrors my_functions.keypoint_extraction so the same canonical hand layout
// reaches the trained model regardless of which physical hand the user signs with.

const LANDMARKS_PER_HAND = 21;
const COORDS_PER_LANDMARK = 3;
const HAND_VECTOR_LENGTH = LANDMARKS_PER_HAND * COORDS_PER_LANDMARK; // 63

export function zeros(n) {
  return new Array(n).fill(0);
}

function l2(dx, dy, dz) {
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function normalizeHand(landmarks) {
  if (!landmarks || landmarks.length !== LANDMARKS_PER_HAND) {
    return zeros(HAND_VECTOR_LENGTH);
  }
  const wrist = landmarks[0];
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  for (const p of landmarks) {
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.z < minZ) minZ = p.z;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
    if (p.z > maxZ) maxZ = p.z;
  }
  const spread = l2(maxX - minX, maxY - minY, maxZ - minZ);
  const out = new Array(HAND_VECTOR_LENGTH);
  for (let i = 0; i < LANDMARKS_PER_HAND; i++) {
    const dx = landmarks[i].x - wrist.x;
    const dy = landmarks[i].y - wrist.y;
    const dz = landmarks[i].z - wrist.z;
    const o = i * COORDS_PER_LANDMARK;
    if (spread > 0) {
      out[o] = dx / spread;
      out[o + 1] = dy / spread;
      out[o + 2] = dz / spread;
    } else {
      out[o] = dx;
      out[o + 1] = dy;
      out[o + 2] = dz;
    }
  }
  return out;
}

function mirrorHand(landmarks) {
  if (!landmarks) return zeros(HAND_VECTOR_LENGTH);
  return normalizeHand(landmarks.map((p) => ({ x: -p.x, y: p.y, z: p.z })));
}

function isLeftHandHeuristic(landmarks) {
  // Same heuristic as my_functions.is_left_hand: thumb tip x > index tip x
  if (!landmarks) return false;
  return landmarks[4].x > landmarks[8].x;
}

/**
 * Build the (126,) keypoint frame from up to two detected hands.
 * leftHand / rightHand follow MediaPipe's classification (after we mirror the
 * frame), and the algorithm corrects with the thumb/index heuristic.
 */
export function extractKeypoints({ leftHand, rightHand }) {
  let lh;
  let rh;
  if (leftHand && rightHand) {
    const leftIsLeft = isLeftHandHeuristic(leftHand);
    const rightIsLeft = isLeftHandHeuristic(rightHand);
    if (leftIsLeft && !rightIsLeft) {
      lh = normalizeHand(leftHand);
      rh = mirrorHand(rightHand);
    } else if (rightIsLeft && !leftIsLeft) {
      lh = normalizeHand(rightHand);
      rh = mirrorHand(leftHand);
    } else {
      lh = normalizeHand(leftHand);
      rh = zeros(HAND_VECTOR_LENGTH);
    }
  } else if (leftHand) {
    lh = isLeftHandHeuristic(leftHand) ? normalizeHand(leftHand) : mirrorHand(leftHand);
    rh = zeros(HAND_VECTOR_LENGTH);
  } else if (rightHand) {
    lh = isLeftHandHeuristic(rightHand) ? normalizeHand(rightHand) : mirrorHand(rightHand);
    rh = zeros(HAND_VECTOR_LENGTH);
  } else {
    lh = zeros(HAND_VECTOR_LENGTH);
    rh = zeros(HAND_VECTOR_LENGTH);
  }
  return [...lh, ...rh];
}

export const HAND_CONNECTIONS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [0, 17]
];

export function drawHandLandmarks(ctx, hands) {
  if (!ctx || !hands) return;
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  ctx.save();
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(34, 197, 94, 0.9)";
  ctx.fillStyle = "rgba(239, 68, 68, 0.95)";
  for (const hand of hands) {
    if (!hand) continue;
    for (const [a, b] of HAND_CONNECTIONS) {
      const pa = hand[a];
      const pb = hand[b];
      if (!pa || !pb) continue;
      ctx.beginPath();
      ctx.moveTo(pa.x * w, pa.y * h);
      ctx.lineTo(pb.x * w, pb.y * h);
      ctx.stroke();
    }
    for (const p of hand) {
      ctx.beginPath();
      ctx.arc(p.x * w, p.y * h, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.restore();
}

// --- Hand rendering for text-to-gesture playback -----------------------

function hasHandData(vec) {
  if (!vec || vec.length !== HAND_VECTOR_LENGTH) return false;
  for (let i = 0; i < vec.length; i++) {
    if (Math.abs(vec[i]) > 1e-6) return true;
  }
  return false;
}

/** Target hand bbox height as a fraction of the camera frame (image coords). */
export const RAW_TARGET_HAND_SPAN = 0.3;

export function isRawLandmarkVector(vec) {
  if (!vec || vec.length !== HAND_VECTOR_LENGTH) return false;
  if (!hasHandData(vec)) return false;
  let inImageSpace = 0;
  for (let i = 0; i < LANDMARKS_PER_HAND; i++) {
    const x = vec[i * COORDS_PER_LANDMARK];
    const y = vec[i * COORDS_PER_LANDMARK + 1];
    if (x >= -0.08 && x <= 1.08 && y >= -0.08 && y <= 1.08) inImageSpace++;
  }
  return inImageSpace >= 12;
}

export function isRawPlaybackFrame(lh, rh, playbackFormat) {
  if (playbackFormat === "raw_v1") return true;
  return isRawLandmarkVector(lh) || isRawLandmarkVector(rh);
}

/**
 * Scale finger geometry relative to the wrist so every gesture appears at a
 * similar hand size, while keeping the wrist position untouched (preserves
 * whole-hand translation like ц top-to-bottom).
 */
export function standardizeRawHand(vec, targetSpan = RAW_TARGET_HAND_SPAN) {
  if (!hasHandData(vec)) return vec;
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < LANDMARKS_PER_HAND; i++) {
    const x = vec[i * COORDS_PER_LANDMARK];
    const y = vec[i * COORDS_PER_LANDMARK + 1];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const span = Math.max(maxX - minX, maxY - minY, 1e-6);
  const scale = targetSpan / span;
  if (Math.abs(scale - 1) < 0.02) return vec;

  const wx = vec[0];
  const wy = vec[1];
  const out = vec.slice();
  for (let i = 0; i < LANDMARKS_PER_HAND; i++) {
    const o = i * COORDS_PER_LANDMARK;
    out[o] = wx + (vec[o] - wx) * scale;
    out[o + 1] = wy + (vec[o + 1] - wy) * scale;
  }
  return out;
}

/** Standardize every frame in a raw playback clip for display. */
export function prepareRawPlaybackSequence(sequence, targetSpan = RAW_TARGET_HAND_SPAN) {
  if (!Array.isArray(sequence)) return [];
  return sequence.map((frame) => ({
    lh: standardizeRawHand(frame?.lh, targetSpan),
    rh: standardizeRawHand(frame?.rh, targetSpan),
  }));
}

/**
 * Compute the data-space bounding box of every hand across a 20-frame sequence.
 * Works for raw image coords (0..1) and legacy wrist-centered vectors.
 */
export function computeSequenceBounds(sequence) {
  if (!Array.isArray(sequence) || sequence.length === 0) return null;
  const collect = (key) => {
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    let any = false;
    for (const frame of sequence) {
      const v = frame ? frame[key] : null;
      if (!hasHandData(v)) continue;
      any = true;
      for (let i = 0; i < LANDMARKS_PER_HAND; i++) {
        const x = v[i * 3];
        const y = v[i * 3 + 1];
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
    return any ? { minX, maxX, minY, maxY } : null;
  };
  const lh = collect("lh");
  const rh = collect("rh");
  if (!lh && !rh) return null;
  return { lh, rh };
}

/** Union of per-hand bounds — used for raw two-hand clips in image space. */
function mergeHandBounds(lhBounds, rhBounds) {
  if (!lhBounds && !rhBounds) return null;
  if (!lhBounds) return rhBounds;
  if (!rhBounds) return lhBounds;
  return {
    minX: Math.min(lhBounds.minX, rhBounds.minX),
    maxX: Math.max(lhBounds.maxX, rhBounds.maxX),
    minY: Math.min(lhBounds.minY, rhBounds.minY),
    maxY: Math.max(lhBounds.maxY, rhBounds.maxY),
  };
}

// Convert a 63-float wrist-centered vector into 21 (x,y) screen points,
// auto-fitting into the requested box. The `margin` parameter is the
// fraction of the box reserved as padding on each axis (smaller -> larger
// hand). When `bounds` is provided, those data-space extremes are used
// instead of the current frame's own min/max, which is essential for
// animations: every frame shares the same transform so motion is preserved.
function buildScreenPoints(
  vec,
  boxX,
  boxY,
  boxW,
  boxH,
  mirrorX = false,
  margin = 0.18,
  bounds = null
) {
  let minX;
  let maxX;
  let minY;
  let maxY;
  if (bounds) {
    // `bounds` is stored in raw data space; flip if we're mirroring the hand.
    minX = mirrorX ? -bounds.maxX : bounds.minX;
    maxX = mirrorX ? -bounds.minX : bounds.maxX;
    minY = bounds.minY;
    maxY = bounds.maxY;
  } else {
    minX = Infinity;
    maxX = -Infinity;
    minY = Infinity;
    maxY = -Infinity;
    for (let i = 0; i < LANDMARKS_PER_HAND; i++) {
      const x = mirrorX ? -vec[i * 3] : vec[i * 3];
      const y = vec[i * 3 + 1];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }

  const targetW = boxW * (1 - margin);
  const targetH = boxH * (1 - margin);
  const spanX = Math.max(maxX - minX, 1e-6);
  const spanY = Math.max(maxY - minY, 1e-6);
  const scale = Math.min(targetW / spanX, targetH / spanY);

  const handW = spanX * scale;
  const handH = spanY * scale;
  const offsetX = boxX + (boxW - handW) / 2 - minX * scale;
  const offsetY = boxY + (boxH - handH) / 2 - minY * scale;

  const points = new Array(LANDMARKS_PER_HAND);
  for (let i = 0; i < LANDMARKS_PER_HAND; i++) {
    const x = mirrorX ? -vec[i * 3] : vec[i * 3];
    const y = vec[i * 3 + 1];
    points[i] = { x: x * scale + offsetX, y: y * scale + offsetY };
  }
  return points;
}

function strokeHand(ctx, points, palette) {
  ctx.save();
  ctx.lineWidth = palette.lineWidth ?? 2.4;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = palette.stroke;
  ctx.fillStyle = palette.fill;
  for (const [a, b] of HAND_CONNECTIONS) {
    const pa = points[a];
    const pb = points[b];
    if (!pa || !pb) continue;
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
  }
  for (const p of points) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, palette.dotRadius ?? 3, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

const HAND_PALETTE = {
  stroke: "rgba(96, 165, 250, 0.95)",
  fill: "rgba(248, 250, 252, 0.95)"
};

/**
 * Render one canonical gesture frame onto a canvas using the lexicon vectors.
 * Both `lh` and `rh` are 63-float arrays (or null/undefined). The function
 * fits whichever hand(s) are present into the canvas, side-by-side when both
 * are provided.
 */
export function drawGestureFrame(canvas, lh, rh, options = {}) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const w = canvas.width;
  const h = canvas.height;

  ctx.fillStyle = options.background ?? "#0f172a";
  ctx.fillRect(0, 0, w, h);

  const playbackFormat = options.playbackFormat ?? null;
  const isRaw =
    options.coordinateSpace === "raw" ||
    isRawPlaybackFrame(lh, rh, playbackFormat);

  let drawLh = lh;
  let drawRh = rh;
  if (isRaw) {
    const targetSpan = options.targetHandSpan ?? RAW_TARGET_HAND_SPAN;
    drawLh = standardizeRawHand(lh, targetSpan);
    drawRh = standardizeRawHand(rh, targetSpan);
  }

  const hasLh = hasHandData(drawLh);
  const hasRh = hasHandData(drawRh);

  if (!hasLh && !hasRh) {
    ctx.fillStyle = "rgba(148, 163, 184, 0.85)";
    ctx.font = `600 ${Math.max(16, Math.min(w, h) / 18)}px Arial, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(options.emptyLabel ?? "no hand", w / 2, h / 2);
    return;
  }

  const minSide = Math.min(w, h);
  const palette = {
    ...HAND_PALETTE,
    lineWidth: options.lineWidth ?? Math.max(2.5, minSide / 80),
    dotRadius: options.dotRadius ?? Math.max(3, minSide / 60),
  };
  const padding = options.padding ?? (isRaw ? 0.05 : 0.18);
  const lhBounds = options.bounds?.lh ?? null;
  const rhBounds = options.bounds?.rh ?? null;
  const mirrorSingleHand = !isRaw;

  if (hasLh && hasRh) {
    // Raw clips store both hands in shared image coordinates (0..1). Draw them
    // on the full canvas so they stay where they were recorded. Splitting lh
    // into the left half and rh into the right half swaps sides on mirror-view
    // footage (anatomical lh often sits screen-right, rh screen-left).
    if (isRaw) {
      const merged = mergeHandBounds(lhBounds, rhBounds);
      strokeHand(
        ctx,
        buildScreenPoints(drawLh, 0, 0, w, h, false, padding, merged),
        palette
      );
      strokeHand(
        ctx,
        buildScreenPoints(drawRh, 0, 0, w, h, false, padding, merged),
        palette
      );
      return;
    }

    const halfW = w / 2;
    strokeHand(
      ctx,
      buildScreenPoints(drawLh, 0, 0, halfW, h, false, padding, lhBounds),
      palette
    );
    strokeHand(
      ctx,
      buildScreenPoints(drawRh, halfW, 0, halfW, h, true, padding, rhBounds),
      palette
    );
    return;
  }

  const vec = hasLh ? drawLh : drawRh;
  const handBounds = hasLh ? lhBounds : rhBounds;
  strokeHand(
    ctx,
    buildScreenPoints(vec, 0, 0, w, h, mirrorSingleHand, padding, handBounds),
    palette
  );
}
