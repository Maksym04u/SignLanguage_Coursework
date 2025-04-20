# %%

# Import necessary libraries
import os
import numpy as np
import cv2
import mediapipe as mp
from itertools import product
from my_functions import *
import logging
import time
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SignLanguageDataCollector:
    def __init__(self, actions: List[str], sequences: int = 30, frames: int = 10):
        self.actions = np.array(actions)
        self.sequences = sequences
        self.frames = frames
        self.PATH = os.path.join('data')
        self.detection_area = None
        self.setup_directories()
        
    def setup_directories(self) -> None:
        """Create necessary directories for data storage"""
        try:
            for action, sequence in product(self.actions, range(self.sequences)):
                os.makedirs(os.path.join(self.PATH, action, str(sequence)), exist_ok=True)
            logging.info("Directories created successfully")
        except Exception as e:
            logging.error(f"Error creating directories: {e}")
            raise

    def setup_detection_area(self, frame_width: int, frame_height: int) -> None:
        """Setup the detection area on the left side of the frame"""
        # Position on the left side, taking up about 60% of the frame height
        area_size = int(frame_height * 0.6)  # Larger area
        margin = int(frame_height * 0.1)  # 10% margin from the top
        
        self.detection_area = {
            'x1': 50,  # Fixed margin from left edge
            'y1': margin,
            'x2': 50 + area_size,  # Square area
            'y2': margin + area_size
        }

    def draw_detection_area(self, image: np.ndarray, recording: bool = False) -> None:
        """Draw the detection area and guidance on the image"""
        if self.detection_area is None:
            return
            
        # Draw the detection area rectangle
        color = (0, 255, 0) if not recording else (0, 0, 255)  # Green when waiting, Red when recording
        thickness = 2 if not recording else 3  # Thicker border when recording
        
        cv2.rectangle(image, 
                     (self.detection_area['x1'], self.detection_area['y1']),
                     (self.detection_area['x2'], self.detection_area['y2']),
                     color, thickness)
        
        # Add guidance text
        text = 'Perform sign here' if not recording else 'Recording...'
        cv2.putText(image, text,
                    (self.detection_area['x1'], self.detection_area['y1'] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def initialize_camera(self) -> cv2.VideoCapture:
        """Initialize camera with basic error handling"""
        logging.info("Initializing camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot access camera")
        return cap

    def collect_data(self) -> None:
        """Data collection loop with improved interface"""
        cap = self.initialize_camera()
        
        # Get frame dimensions and setup detection area
        ret, frame = cap.read()
        if ret:
            self.setup_detection_area(frame.shape[1], frame.shape[0])
        
        try:
            with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
                for action, sequence in product(self.actions, range(self.sequences)):
                    print(f"\nRecording {action}, sequence {sequence + 1}/{self.sequences}")
                    print("Press SPACE to start recording, 'q' to quit")
                    
                    # Wait for space key
                    while True:
                        ret, image = cap.read()
                        if not ret:
                            continue
                            
                        results = image_process(image, holistic)
                        draw_landmarks(image, results)
                        self.draw_detection_area(image)
                        
                        # Display instructions
                        cv2.putText(image, f'Action: {action}', (20, 20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.putText(image, f'Sequence: {sequence + 1}/{self.sequences}', (20, 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.putText(image, 'Press SPACE to start', (20, 400), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        
                        cv2.imshow('Camera', image)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord(' '):
                            break
                        elif key == ord('q'):
                            raise KeyboardInterrupt
                            
                    # Record sequence
                    print("Recording...")
                    for frame in range(self.frames):
                        ret, image = cap.read()
                        if not ret:
                            continue
                            
                        results = image_process(image, holistic)
                        draw_landmarks(image, results)
                        self.draw_detection_area(image, recording=True)
                        
                        # Save keypoints
                        keypoints = keypoint_extraction(results)
                        frame_path = os.path.join(self.PATH, action, str(sequence), str(frame))
                        np.save(frame_path, keypoints)
                        
                        # Display progress
                        cv2.putText(image, f'Frame: {frame + 1}/{self.frames}', (20, 20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('Camera', image)
                        cv2.waitKey(1)
                        
                    print("Sequence completed")
                    
        except KeyboardInterrupt:
            print("\nData collection interrupted by user")
        except Exception as e:
            logging.error(f"Error during data collection: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Define the actions (signs) that will be recorded
    actions = ['a', 'b', 'c', 'd']
    
    # Create collector instance and start data collection
    collector = SignLanguageDataCollector(actions)
    collector.collect_data()
