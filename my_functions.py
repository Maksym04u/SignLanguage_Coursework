import os
import sys
from pathlib import Path

import mediapipe as mp
import cv2
import numpy as np

_UI_FONT_CACHE = {}


def _unicode_font_path() -> Path | None:
    """Return a TrueType font that supports Cyrillic (Windows / Linux)."""
    if sys.platform == "win32":
        fonts_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        candidates = [
            fonts_dir / "segoeui.ttf",
            fonts_dir / "arial.ttf",
            fonts_dir / "Tahoma.ttf",
        ]
    else:
        candidates = [
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def put_ui_text(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    font_scale: float = 0.5,
    color_bgr: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 1,
) -> np.ndarray:
    """
    Draw text on a BGR image. Uses a Unicode-capable font so Ukrainian letters
    render correctly (cv2.putText only supports ASCII and shows '??' otherwise).
    """
    if not text:
        return image

    font_size = max(12, int(round(font_scale * 32)))
    font_path = _unicode_font_path()

    if font_path is None:
        cv2.putText(
            image,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color_bgr,
            thickness,
            cv2.LINE_AA,
        )
        return image

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        cv2.putText(
            image,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color_bgr,
            thickness,
            cv2.LINE_AA,
        )
        return image

    cache_key = (str(font_path), font_size)
    if cache_key not in _UI_FONT_CACHE:
        _UI_FONT_CACHE[cache_key] = ImageFont.truetype(str(font_path), font_size)

    font = _UI_FONT_CACHE[cache_key]
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    x, y = org
    # Pillow anchors from top-left; nudge down so it lines up with cv2 baseline.
    draw.text((x, y), text, font=font, fill=rgb)
    image[:, :] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return image

def draw_landmarks(image, results):
    """
    Draw the landmarks on the image.

    Args:
        image (numpy.ndarray): The input image.
        results: The landmarks detected by Mediapipe.

    Returns:
        None
    """
    # Draw landmarks for left hand
    mp.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
    # Draw landmarks for right hand
    mp.solutions.drawing_utils.draw_landmarks(image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)

def image_process(image, model):
    """
    Process the image and obtain sign landmarks.

    Args:
        image (numpy.ndarray): The input image.
        model: The Mediapipe holistic object.

    Returns:
        results: The processed results containing sign landmarks.
    """
    # Set the image to read-only mode
    image.flags.writeable = True
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Process the image using the model
    results = model.process(image)
    # Set the image back to writeable mode
    image.flags.writeable = True
    # Convert the image back from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return results

def keypoint_extraction(results):
    """
    Extract and normalize the keypoints from the sign landmarks.
    For left hand, keep landmarks as is.
    For right hand, identify and adjust landmarks to match left hand format.

    Args:
        results: The processed results containing sign landmarks.

    Returns:
        keypoints (numpy.ndarray): The normalized keypoints.
    """
    def normalize_hand_landmarks(landmarks):
        if not landmarks:
            return np.zeros(63)
            
        # Convert landmarks to numpy array
        points = np.array([[res.x, res.y, res.z] for res in landmarks.landmark])
        
        # Get wrist position (first landmark)
        wrist = points[0]
        
        # Normalize by subtracting wrist position (relative to wrist)
        normalized = points - wrist
        
        # Scale using hand spread (bounding box size)
        hand_spread = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
        if hand_spread > 0:  # Avoid division by zero
            normalized = normalized / hand_spread
            
        return normalized.flatten()
    
    def is_left_hand(landmarks):
        if not landmarks:
            return False
            
        # Get thumb tip (landmark 4) and index finger tip (landmark 8)
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        
        # For left hand, thumb tip should be to the right of index tip
        return thumb_tip.x > index_tip.x
    
    def mirror_landmarks(landmarks):
        """Mirror the landmarks horizontally to convert right hand to left hand format"""
        if not landmarks:
            return np.zeros(63)
            
        # Convert landmarks to numpy array
        points = np.array([[res.x, res.y, res.z] for res in landmarks.landmark])
        
        # Mirror the x coordinates (multiply by -1)
        points[:, 0] = -points[:, 0]
        
        # Get wrist position (first landmark)
        wrist = points[0]
        
        # Normalize by subtracting wrist position (relative to wrist)
        normalized = points - wrist
        
        # Scale using hand spread (bounding box size)
        hand_spread = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
        if hand_spread > 0:  # Avoid division by zero
            normalized = normalized / hand_spread
            
        return normalized.flatten()
    
    # Check which hand is being used
    if results.left_hand_landmarks and results.right_hand_landmarks:
        # Both hands detected, determine which is which
        left_is_left = is_left_hand(results.left_hand_landmarks)
        right_is_left = is_left_hand(results.right_hand_landmarks)
        
        if left_is_left and not right_is_left:
            # Left hand is correctly identified, right hand is right
            lh = normalize_hand_landmarks(results.left_hand_landmarks)
            rh = mirror_landmarks(results.right_hand_landmarks)
        elif right_is_left and not left_is_left:
            # Right hand is actually left hand, left hand is right
            lh = normalize_hand_landmarks(results.right_hand_landmarks)
            rh = mirror_landmarks(results.left_hand_landmarks)
        else:
            # If both are detected as same type, use the one that's more visible
            lh = normalize_hand_landmarks(results.left_hand_landmarks)
            rh = np.zeros(63)
    elif results.left_hand_landmarks:
        # Only left hand detected, verify it's actually left
        if is_left_hand(results.left_hand_landmarks):
            lh = normalize_hand_landmarks(results.left_hand_landmarks)
            rh = np.zeros(63)
        else:
            # It's actually a right hand, mirror it
            lh = mirror_landmarks(results.left_hand_landmarks)
            rh = np.zeros(63)
    elif results.right_hand_landmarks:
        # Only right hand detected, verify it's actually right
        if not is_left_hand(results.right_hand_landmarks):
            lh = mirror_landmarks(results.right_hand_landmarks)
            rh = np.zeros(63)
        else:
            # It's actually a left hand
            lh = normalize_hand_landmarks(results.right_hand_landmarks)
            rh = np.zeros(63)
    else:
        # No hands detected
        lh = np.zeros(63)
        rh = np.zeros(63)
    
    # Concatenate the normalized keypoints for both hands
    keypoints = np.concatenate([lh, rh])
    return keypoints
