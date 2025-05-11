# %%

# Import necessary libraries
import numpy as np
import os
import string
import mediapipe as mp
import cv2
from my_functions import *
import keyboard
from keras._tf_keras.keras.models import load_model
import language_tool_python
import time
import logging
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SignLanguageTranslator:
    def __init__(self):
        # Set the path to the data directory
        self.PATH = os.path.join('data')
        
        # Create an array of action labels
        self.actions = np.array(os.listdir(self.PATH))
        
        # Initialize lists
        self.sentence = []
        self.keypoints_buffer = deque(maxlen=20)  # Buffer to store the last 20 frames
        self.last_prediction = None
        self.grammar_result = None
        self.current_confidence = 0.0
        self.current_prediction = None
        
        # Performance tracking
        self.frame_count = 0
        self.last_prediction_time = 0
        self.prediction_interval = 0.7  # Make predictions every 0.5 seconds
        
        # Buffer reset mechanism
        self.last_landmark_time = 0
        self.buffer_timeout = 1.0  # Reset buffer after 2 seconds of no landmarks
        
        # Recognition state
        self.recognition_in_progress = False
        self.recognition_start_time = 0
        self.recognition_duration = 1.0  # Duration to show recognition message
        
        # Buffer reset delay
        self.buffer_reset_delay = 0.75  # Wait 0.5 seconds before resetting buffer after recognition
        self.buffer_reset_time = 0  # Time when buffer should be reset
        
        # Load model with error handling
        try:
            self.model = load_model('my_model.h5')
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise RuntimeError("Could not load model. Make sure you have trained the model first.")
            
        # Initialize grammar tool
        try:
            self.tool = language_tool_python.LanguageToolPublicAPI('en-UK')
        except Exception as e:
            logging.error(f"Error initializing grammar tool: {e}")
            self.tool = None

    def draw_confidence_bar(self, image: np.ndarray) -> None:
        """Draw a confidence bar showing the current prediction confidence"""
        bar_width = 200
        bar_height = 20
        x = 20
        y = 100
        
        # Draw background bar
        cv2.rectangle(image, (x, y), (x + bar_width, y + bar_height), (128, 128, 128), -1)
        
        # Draw confidence level
        confidence_width = int(bar_width * self.current_confidence)
        color = (0, 255, 0) if self.current_confidence >= 0.98 else (0, 165, 255)  # Green if confident, orange if not
        cv2.rectangle(image, (x, y), (x + confidence_width, y + bar_height), color, -1)
        
        # Draw border
        cv2.rectangle(image, (x, y), (x + bar_width, y + bar_height), (255, 255, 255), 1)
        
        # Draw text
        confidence_text = f'Confidence: {self.current_confidence:.2%}'
        cv2.putText(image, confidence_text, (x + 5, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
    def process_prediction(self, prediction: np.ndarray) -> str:
        """Process model prediction and return the predicted sign"""
        self.current_confidence = np.amax(prediction)
        self.current_prediction = self.actions[np.argmax(prediction)]
        
        # Check if we have a high confidence prediction (threshold at 98%)
        if self.current_confidence >= 0.98:
            # Only return the prediction if it's different from the last one
            if self.last_prediction != self.current_prediction:
                return self.current_prediction
                
        return None
        
    def update_sentence(self, new_sign: str) -> None:
        """Update the sentence with the new sign"""
        if new_sign and (not self.last_prediction or self.last_prediction != new_sign):
            self.sentence.append(new_sign)
            self.last_prediction = new_sign
            
            # Capitalize first word
            if len(self.sentence) == 1:
                self.sentence[0] = self.sentence[0].capitalize()
                
            # Combine consecutive letters into words
            if len(self.sentence) >= 2:
                if (self.sentence[-1] in string.ascii_lowercase or self.sentence[-1] in string.ascii_uppercase) and \
                   (self.sentence[-2] in string.ascii_lowercase or self.sentence[-2] in string.ascii_uppercase):
                    self.sentence[-1] = self.sentence[-2] + self.sentence[-1]
                    self.sentence.pop(-2)
                    self.sentence[-1] = self.sentence[-1].capitalize()
            
            # Schedule buffer reset after delay
            self.buffer_reset_time = time.time() + self.buffer_reset_delay
            logging.info(f"Scheduled buffer reset in {self.buffer_reset_delay} seconds after recognizing: {new_sign}")
            
            # Set recognition state
            self.recognition_in_progress = True
            self.recognition_start_time = time.time()
                    
    def check_grammar(self) -> None:
        """Check and correct grammar of the current sentence"""
        if self.tool and self.sentence:
            text = ' '.join(self.sentence)
            self.grammar_result = self.tool.correct(text)
            
    def reset(self) -> None:
        """Reset all state variables"""
        self.sentence = []
        self.keypoints_buffer.clear()  # Clear the buffer
        self.last_prediction = None
        self.grammar_result = None
        self.current_confidence = 0.0
        self.current_prediction = None
        self.last_landmark_time = time.time()  # Reset the landmark time
        self.recognition_in_progress = False
        self.buffer_reset_time = 0  # Reset the buffer reset time
        
    def run(self) -> None:
        """Main translation loop"""
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Cannot access camera")
            return
            
        try:
            with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
                while cap.isOpened():
                    ret, image = cap.read()
                    if not ret:
                        logging.error("Failed to grab frame")
                        continue
                    
                    # Flip the image horizontally before processing
                    image = cv2.flip(image, 1)
                        
                    current_time = time.time()
                    
                    # Check if recognition message should be hidden
                    if self.recognition_in_progress and (current_time - self.recognition_start_time) > self.recognition_duration:
                        self.recognition_in_progress = False
                    
                    # Check if buffer should be reset after recognition delay
                    if self.buffer_reset_time > 0 and current_time >= self.buffer_reset_time:
                        self.keypoints_buffer.clear()
                        logging.info("Buffer reset after recognition delay")
                        self.buffer_reset_time = 0
                    
                    # Check if buffer timeout has occurred
                    if len(self.keypoints_buffer) > 0 and (current_time - self.last_landmark_time) > self.buffer_timeout:
                        logging.info("Buffer timeout - resetting buffer")
                        self.keypoints_buffer.clear()
                        
                    # Process image and get landmarks using the same function as data collection
                    try:
                        results = image_process(image, holistic)
                        
                        # Draw landmarks
                        draw_landmarks(image, results)
                        
                        # Check if hands are detected
                        hands_detected = results.left_hand_landmarks is not None or results.right_hand_landmarks is not None
                        
                        if hands_detected:
                            # Extract and store keypoints
                            keypoints = keypoint_extraction(results)
                            
                            if keypoints is not None:
                                # Update the last landmark time
                                self.last_landmark_time = current_time
                                
                                # Add to the buffer (automatically maintains maxlen=20)
                                self.keypoints_buffer.append(keypoints)
                                
                                # Process when we have enough frames (20 frames = ~1.6 seconds)
                                # and enough time has passed since the last prediction
                                if len(self.keypoints_buffer) == 20 and (current_time - self.last_prediction_time) >= self.prediction_interval:
                                    # Convert buffer to numpy array
                                    keypoints_array = np.array(list(self.keypoints_buffer))
                                    
                                    # Make prediction
                                    prediction = self.model.predict(keypoints_array[np.newaxis, :, :], verbose=0)
                                    self.last_prediction_time = current_time
                                    
                                    # Process prediction
                                    predicted_sign = self.process_prediction(prediction)
                                    if predicted_sign:
                                        self.update_sentence(predicted_sign)
                        else:
                            # No hands detected, display message on screen
                            cv2.putText(image, "No hands detected", (20, 350),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                                    
                    except Exception as e:
                        logging.error(f"Error processing frame: {e}")
                        continue
                    
                    # Draw confidence bar
                    self.draw_confidence_bar(image)
                    
                    # Display current prediction if confidence is above 0.5
                    if self.current_confidence > 0.5:
                        prediction_text = f'Detected: {self.current_prediction}'
                        cv2.putText(image, prediction_text, (20, 200),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    # Display buffer size with color indication
                    buffer_size = len(self.keypoints_buffer)
                    buffer_text = f'Buffer: {buffer_size}/20 frames'
                    
                    # Change color based on buffer status
                    if buffer_size == 20:
                        color = (0, 255, 0)  # Green when full
                        cv2.putText(image, "Buffer full - analyzing gesture", (20, 400),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                    elif buffer_size > 15:
                        color = (0, 255, 255)  # Yellow when almost full
                    else:
                        color = (255, 255, 255)  # White otherwise
                        
                    cv2.putText(image, buffer_text, (20, 250),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                    
                    # Display recognition message if in progress
                    if self.recognition_in_progress:
                        cv2.putText(image, f"Recognized: {self.last_prediction}", (20, 450),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    # Display FPS
                    self.frame_count += 1
                    if current_time - self.last_prediction_time >= 1.0:
                        fps = self.frame_count
                        self.frame_count = 0
                        fps_text = f'FPS: {fps}'
                        cv2.putText(image, fps_text, (20, 300),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    # Handle keyboard inputs
                    if keyboard.is_pressed(' '):
                        self.reset()
                    elif keyboard.is_pressed('enter'):
                        self.check_grammar()
                        
                    # Display current sentence or grammar result
                    text_to_display = self.grammar_result if self.grammar_result else ' '.join(self.sentence)
                    if text_to_display:
                        textsize = cv2.getTextSize(text_to_display, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                        text_X_coord = (image.shape[1] - textsize[0]) // 2
                        cv2.putText(image, text_to_display, (text_X_coord, 470),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Display instructions
                    cv2.putText(image, 'Press SPACE to reset, ENTER for grammar check', (20, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    # Show the image
                    cv2.imshow('Camera', image)
                    
                    # Break loop if window is closed
                    if cv2.waitKey(1) & 0xFF == ord('q') or \
                       cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
                        break
                        
        except Exception as e:
            logging.error(f"Error during translation: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.tool:
                self.tool.close()

if __name__ == "__main__":
    translator = SignLanguageTranslator()
    translator.run()
