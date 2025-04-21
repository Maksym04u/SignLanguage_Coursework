import mediapipe as mp
import cv2
import numpy as np

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
        # This is more stable than wrist-to-fingertip distance
        # as it accounts for the entire hand shape
        hand_spread = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
        if hand_spread > 0:  # Avoid division by zero
            normalized = normalized / hand_spread
            
        return normalized.flatten()
    
    # Extract and normalize keypoints for both hands
    lh = normalize_hand_landmarks(results.left_hand_landmarks)
    rh = normalize_hand_landmarks(results.right_hand_landmarks)
    
    # Concatenate the normalized keypoints for both hands
    keypoints = np.concatenate([lh, rh])
    return keypoints
