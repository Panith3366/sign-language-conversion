import cv2
import numpy as np
import json
import time
from tensorflow.keras.models import model_from_json

# Load saved model
with open("Models/model_new.json", "r") as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
model.load_weights("Models/model_new.h5")
print("Loaded model from disk")

# Class labels
classes = ['0'] + list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# For FPS calculation
prev_time = 0
curr_time = 0

try:
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        # Preprocess frame (same as training)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (128, 128))
        input_img = np.array(resized).reshape(1, 128, 128, 1) / 255.0
        
        # Get prediction
        predictions = model.predict(input_img, verbose=0)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        # Display info
        label = f"Prediction: {classes[predicted_class]} ({confidence*100:.1f}%)"
        fps_text = f"FPS: {fps:.1f}"
        
        cv2.putText(frame, label, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, fps_text, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Add rectangle around hand area
        cv2.rectangle(frame, (50, 50), (300, 300), (255, 0, 0), 2)
        
        # Show frame
        cv2.imshow('Sign Language Recognition', frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released")
