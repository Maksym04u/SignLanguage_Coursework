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
        self.keypoints = []
        self.last_prediction = None
        self.grammar_result = None
        self.detection_area = None
        self.current_confidence = 0.0
        self.current_prediction = None
        
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

    def draw_detection_area(self, image: np.ndarray) -> None:
        """Draw the detection area and guidance on the image"""
        if self.detection_area is None:
            return
            
        # Draw the detection area rectangle
        color = (0, 255, 0)  # Green color
        thickness = 2
        
        cv2.rectangle(image, 
                     (self.detection_area['x1'], self.detection_area['y1']),
                     (self.detection_area['x2'], self.detection_area['y2']),
                     color, thickness)
        
        # Add guidance text
        cv2.putText(image, 'Perform sign here',
                    (self.detection_area['x1'], self.detection_area['y1'] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

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
        color = (0, 255, 0) if self.current_confidence >= 0.9 else (0, 165, 255)  # Green if confident, orange if not
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
        
        # Only accept predictions with high confidence
        if self.current_confidence > 0.98:
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
                    
    def check_grammar(self) -> None:
        """Check and correct grammar of the current sentence"""
        if self.tool and self.sentence:
            text = ' '.join(self.sentence)
            self.grammar_result = self.tool.correct(text)
            
    def reset(self) -> None:
        """Reset all state variables"""
        self.sentence = []
        self.keypoints = []
        self.last_prediction = None
        self.grammar_result = None
        self.current_confidence = 0.0
        self.current_prediction = None
        
    def run(self) -> None:
        """Main translation loop"""
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Cannot access camera")
            return
            
        # Get frame dimensions and setup detection area
        ret, frame = cap.read()
        if ret:
            self.setup_detection_area(frame.shape[1], frame.shape[0])
            
        try:
            with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
                while cap.isOpened():
                    ret, image = cap.read()
                    if not ret:
                        logging.error("Failed to grab frame")
                        continue
                        
                    # Process image and get landmarks using the same function as data collection
                    try:
                        results = image_process(image, holistic)
                        
                        # Draw landmarks
                        draw_landmarks(image, results)
                        
                        # Extract and store keypoints
                        keypoints = keypoint_extraction(results)
                        if keypoints is not None:
                            self.keypoints.append(keypoints)
                            
                            # Process when we have enough frames
                            if len(self.keypoints) == 10:
                                keypoints_array = np.array(self.keypoints)
                                prediction = self.model.predict(keypoints_array[np.newaxis, :, :])
                                self.keypoints = []  # Reset for next sequence
                                
                                # Process prediction
                                predicted_sign = self.process_prediction(prediction)
                                if predicted_sign:
                                    self.update_sentence(predicted_sign)
                                    
                    except Exception as e:
                        logging.error(f"Error processing frame: {e}")
                        continue
                        
                    # Draw detection area
                    self.draw_detection_area(image)
                    
                    # Draw confidence bar
                    self.draw_confidence_bar(image)
                    
                    # Display current prediction if confidence is above 0.5
                    if self.current_confidence > 0.5:
                        prediction_text = f'Detected: {self.current_prediction}'
                        cv2.putText(image, prediction_text, (20, 150),
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
