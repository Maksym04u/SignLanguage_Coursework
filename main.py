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
        self.capitalize_next = False  # Flag to track if next letter should be capitalized
        
        # Window state
        self.is_fullscreen = False
        self.window_name = 'Sign Language Translator'
        
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
        self.buffer_reset_delay = 0.75  # Wait 0.75 seconds before resetting buffer after recognition
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

    def get_scaled_coordinates(self, image, x_percent, y_percent):
        """Convert percentage-based coordinates to actual pixel coordinates"""
        height, width = image.shape[:2]
        return (int(width * x_percent), int(height * y_percent))

    def draw_confidence_bar(self, image: np.ndarray) -> None:
        """Draw a confidence bar showing the current prediction confidence"""
        height, width = image.shape[:2]
        bar_width = int(width * 0.2)  # 20% of screen width
        bar_height = int(height * 0.02)  # 2% of screen height
        
        x, y = self.get_scaled_coordinates(image, 0.02, 0.1)  # 2% from left, 10% from top
        
        # Draw background bar
        cv2.rectangle(image, (x, y), (x + bar_width, y + bar_height), (128, 128, 128), -1)
        
        # Draw confidence level
        confidence_width = int(bar_width * self.current_confidence)
        color = (0, 255, 0) if self.current_confidence >= 0.98 else (0, 165, 255)
        cv2.rectangle(image, (x, y), (x + confidence_width, y + bar_height), color, -1)
        
        # Draw border
        cv2.rectangle(image, (x, y), (x + bar_width, y + bar_height), (255, 255, 255), 1)
        
        # Draw text
        confidence_text = f'Confidence: {self.current_confidence:.2%}'
        font_scale = width / 1000  # Scale font size based on window width
        cv2.putText(image, confidence_text, (x + 5, y + int(bar_height * 0.75)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
            
    def process_prediction(self, prediction: np.ndarray) -> str:
        """Process model prediction and return the predicted sign"""
        self.current_confidence = np.amax(prediction)
        self.current_prediction = self.actions[np.argmax(prediction)]
        
        # Check if we have a high confidence prediction (threshold at 98%)
        if self.current_confidence >= 0.98:
            # Only return the prediction if it's different from the last one
            if (self.last_prediction != self.current_prediction or self.current_prediction not in string.ascii_lowercase or self.current_prediction not in string.ascii_uppercase):
                return self.current_prediction
                
        return None
        
    def update_sentence(self, new_sign: str) -> None:
        """Update the sentence with the new sign"""
        if new_sign:
            # Only check for duplicates if the last prediction exists and is not a letter
            if self.last_prediction and new_sign == self.last_prediction and \
               not (new_sign in string.ascii_lowercase or new_sign in string.ascii_uppercase):
                return
                
            # Convert letter to lowercase by default
            if new_sign in string.ascii_uppercase:
                new_sign = new_sign.lower()
                
            self.sentence.append(new_sign)
            self.last_prediction = new_sign
            
            # Capitalize first word
            if len(self.sentence) == 1:
                self.sentence[0] = self.sentence[0].capitalize()
                
            # Combine consecutive letters into words without spaces
            if len(self.sentence) >= 2:
                if (self.sentence[-1] in string.ascii_lowercase or self.sentence[-1] in string.ascii_uppercase) and \
                   (self.sentence[-2] in string.ascii_lowercase or self.sentence[-2] in string.ascii_uppercase):
                    self.sentence[-1] = self.sentence[-2] + self.sentence[-1]
                    self.sentence.pop(-2)
            
            # Clear buffer immediately after recognizing any gesture
            self.keypoints_buffer.clear()
            # Set delay for next recognition
            self.buffer_reset_time = time.time() + self.buffer_reset_delay
            logging.info(f"Buffer cleared and next recognition scheduled in {self.buffer_reset_delay} seconds after recognizing: {new_sign}")
            
            # Set recognition state
            self.recognition_in_progress = True
            self.recognition_start_time = time.time()
            
    def add_space(self) -> None:
        """Add a space to the current sentence and capitalize the next letter"""
        if self.sentence:
            self.sentence.append(" ")
            logging.info("Space added to sentence")
            # Set a flag to capitalize the next letter
            self.capitalize_next = True
            
    def check_grammar(self) -> None:
        """Check and correct grammar of the current sentence"""
        if self.tool and self.sentence:
            text = ''.join(self.sentence)  # Join without spaces
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
            
        # Set initial window size
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1920, 1080)  # Wider window to accommodate split view
            
        try:
            with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
                while cap.isOpened():
                    ret, image = cap.read()
                    if not ret:
                        logging.error("Failed to grab frame")
                        continue
                    
                    # Flip the image horizontally before processing
                    image = cv2.flip(image, 1)
                    
                    # Create a blank canvas for the split view
                    height, width = image.shape[:2]
                    # Create a wider canvas to accommodate both views
                    combined_width = width + 400  # Add 400 pixels for text area
                    combined_image = np.zeros((height, combined_width, 3), dtype=np.uint8)
                    
                    # Place camera feed on the left side
                    combined_image[:, :width] = image
                    
                    # Create text area on the right side (white background)
                    text_area = np.ones((height, 400, 3), dtype=np.uint8) * 255
                    combined_image[:, width:] = text_area
                        
                    current_time = time.time()
                    
                    # Get current window size
                    height, width = image.shape[:2]
                    font_scale = width / 1000  # Scale font size based on window width
                    
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
                        
                        # Draw landmarks on the combined image (left side)
                        draw_landmarks(combined_image[:, :width], results)
                        
                        # Check if hands are detected
                        hands_detected = results.left_hand_landmarks is not None or results.right_hand_landmarks is not None
                        
                        if hands_detected:
                            # Extract and store keypoints
                            keypoints = keypoint_extraction(results)
                            
                            if keypoints is not None:
                                # Update the last landmark time
                                self.last_landmark_time = current_time
                                
                                # Only add to buffer if we're not in the reset delay period
                                if self.buffer_reset_time == 0:
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
                            x, y = self.get_scaled_coordinates(image, 0.02, 0.35)
                            cv2.putText(combined_image, "No hands detected", (x, y),
                                      cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2, cv2.LINE_AA)
                                    
                    except Exception as e:
                        logging.error(f"Error processing frame: {e}")
                        continue
                    
                    # Draw confidence bar
                    self.draw_confidence_bar(combined_image)
                    
                    # Display current prediction if confidence is above 0.5
                    if self.current_confidence > 0.5:
                        x, y = self.get_scaled_coordinates(image, 0.02, 0.2)
                        prediction_text = f'Detected: {self.current_prediction}'
                        cv2.putText(combined_image, prediction_text, (x, y),
                                  cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    # Display buffer size with color indication
                    buffer_size = len(self.keypoints_buffer)
                    buffer_text = f'Buffer: {buffer_size}/20 frames'
                    
                    # Change color based on buffer status
                    if buffer_size == 20:
                        color = (0, 255, 0)  # Green when full
                        x, y = self.get_scaled_coordinates(image, 0.02, 0.4)
                        cv2.putText(combined_image, "Buffer full - analyzing gesture", (x, y),
                                  cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)
                    elif buffer_size > 15:
                        color = (0, 255, 255)  # Yellow when almost full
                    else:
                        color = (255, 255, 255)  # White otherwise
                        
                    x, y = self.get_scaled_coordinates(image, 0.02, 0.25)
                    cv2.putText(combined_image, buffer_text, (x, y),
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
                    
                    # Display recognition message if in progress
                    if self.recognition_in_progress:
                        x, y = self.get_scaled_coordinates(image, 0.02, 0.45)
                        cv2.putText(combined_image, f"Recognized: {self.last_prediction}", (x, y),
                                  cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    # Display FPS
                    self.frame_count += 1
                    if current_time - self.last_prediction_time >= 1.0:
                        fps = self.frame_count
                        self.frame_count = 0
                        fps_text = f'FPS: {fps}'
                        x, y = self.get_scaled_coordinates(image, 0.02, 0.3)
                        cv2.putText(combined_image, fps_text, (x, y),
                                  cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    # Handle keyboard inputs
                    if keyboard.is_pressed(' '):
                        self.add_space()  # Add space instead of resetting
                    elif keyboard.is_pressed('r'):  # Use 'r' for reset
                        self.reset()
                    elif keyboard.is_pressed('enter'):
                        self.check_grammar()
                    elif keyboard.is_pressed('f'):  # Toggle fullscreen
                        self.is_fullscreen = not self.is_fullscreen
                        if self.is_fullscreen:
                            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        else:
                            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    
                    # Display text in the right panel
                    text_to_display = self.grammar_result if self.grammar_result else ''.join(self.sentence)  # Join without spaces
                    if text_to_display:
                        # Calculate maximum characters per line based on panel width
                        max_chars_per_line = 25  # Reduced from 30 to ensure better wrapping
                        
                        # Split text into lines without considering spaces
                        lines = []
                        for i in range(0, len(text_to_display), max_chars_per_line):
                            lines.append(text_to_display[i:i + max_chars_per_line])
                        
                        # Display each line with proper spacing
                        y_offset = height // 4  # Start from 1/4 of the height
                        line_height = 40  # Reduced line height for better spacing
                        
                        for line in lines:
                            # Calculate text size for proper centering
                            textsize = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                            text_x = width + (400 - textsize[0]) // 2  # Center text in the right panel
                            
                            # Draw text with a slight shadow for better readability
                            cv2.putText(combined_image, line, (text_x + 1, y_offset + 1),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2, cv2.LINE_AA)
                            cv2.putText(combined_image, line, (text_x, y_offset),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
                            
                            y_offset += line_height
                            
                            # If we're running out of space, start from the top again
                            if y_offset > height - 50:
                                y_offset = height // 4
                    
                    # Display instructions
                    instructions = 'Press SPACE to add space, R to reset, ENTER for grammar check, F for fullscreen'
                    x, y = self.get_scaled_coordinates(image, 0.02, 0.03)
                    cv2.putText(combined_image, instructions, (x, y),
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    # Draw a vertical line to separate the views
                    cv2.line(combined_image, (width, 0), (width, height), (255, 255, 255), 2)
                    
                    # Show the combined image
                    cv2.imshow(self.window_name, combined_image)
                    
                    # Break loop if window is closed
                    if cv2.waitKey(1) & 0xFF == ord('q') or \
                       cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
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
