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
    def __init__(self, actions: List[str], sequences: int = 30, frames: int = 20):
        self.actions = np.array(actions)
        self.sequences = sequences
        self.frames = frames
        self.PATH = os.path.join('data')
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
        
        try:
            with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
                for action, sequence in product(self.actions, range(self.sequences)):
                    print(f"\nRecording {action}, sequence {sequence + 1}/{self.sequences}")
                    print("Press SPACE to start recording, 'q' to quit")
                    print("You can use either your left or right hand - the system will handle it correctly")
                    
                    # Wait for space key
                    while True:
                        ret, image = cap.read()
                        if not ret:
                            continue
                            
                        # Flip the image horizontally before processing
                        image = cv2.flip(image, 1)
                            
                        results = image_process(image, holistic)
                        draw_landmarks(image, results)
                        
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
                            
                        # Flip the image horizontally before processing
                        image = cv2.flip(image, 1)
                            
                        results = image_process(image, holistic)
                        draw_landmarks(image, results)
                        
                        # Save keypoints using the same function as the translator
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
    actions = ['m', 'n', 'o', 'p', 'q', 'r', 's']
    
    # Create collector instance and start data collection
    collector = SignLanguageDataCollector(actions)
    collector.collect_data()
