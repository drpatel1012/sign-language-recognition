import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import time

# Options
DATA_DIR = './data'
ALPHABET = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
NUM_SAMPLES_PER_SIGN = 50 
CSV_FILE = 'hand_landmarks_az.csv'

# Initialize MediaPipe Hands Task API
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

def draw_landmarks_on_image(rgb_image, hand_landmarks_list):
    annotated_image = np.copy(rgb_image)
    h, w, c = annotated_image.shape
    
    for hand_landmarks in hand_landmarks_list:
        # Draw red circles for landmarks manually instead of using mp.solutions
        for landmark in hand_landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(annotated_image, (x, y), 3, (0, 0, 255), -1)
            
    return annotated_image

def get_landmarks(image):
    # Convert image format for MediaPipe Task API
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    
    # Detect
    detection_result = detector.detect(mp_image)
    
    if detection_result.hand_landmarks:
        hand_landmarks = detection_result.hand_landmarks[0]
        
        # Calculate bounding box for normalization (simple min/max)
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = float('-inf'), float('-inf')
        
        for lm in hand_landmarks:
            x_min = min(x_min, lm.x)
            y_min = min(y_min, lm.y)
            x_max = max(x_max, lm.x)
            y_max = max(y_max, lm.y)
            
        width = x_max - x_min
        height = y_max - y_min
        
        # Normalize landmarks relative to the bounding box
        normalized_landmarks = []
        for lm in hand_landmarks:
            nx = (lm.x - x_min) / (width + 1e-6)
            ny = (lm.y - y_min) / (height + 1e-6)
            nz = lm.z 
            normalized_landmarks.extend([nx, ny, nz])
            
        # Manually draw landmarks
        annotated_image = draw_landmarks_on_image(rgb_image, [hand_landmarks])
        annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            
        return normalized_landmarks, annotated_bgr
    return None, image

def collect_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # To write to CSV
    f = open(os.path.join(DATA_DIR, CSV_FILE), 'w')
    
    # Write Header
    header = ['label']
    for i in range(21): # 21 hand landmarks
        header.extend([f'x{i}', f'y{i}', f'z{i}'])
    f.write(','.join(header) + '\n')
    
    # Wait for user to get ready
    print("Press 'q' to quit at any time.")
    print("Starting data collection in 3 seconds...")
    time.sleep(3)
    
    for label in ALPHABET:
        print(f"\n--- Get ready for sign: {label} ---")
        print("Press 's' to start recording this sign, 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Ready for {label}? Press 's' to start.", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1)
            if key & 0xFF == ord('s'):
                break
            elif key & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                f.close()
                return
                
        # Start recording
        samples_collected = 0
        while samples_collected < NUM_SAMPLES_PER_SIGN:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            landmarks, annotated_frame = get_landmarks(frame)
            
            if landmarks:
                # Save to file
                row = [label] + [str(v) for v in landmarks]
                f.write(','.join(row) + '\n')
                samples_collected += 1
                
            cv2.putText(annotated_frame, f"Recording {label}: {samples_collected}/{NUM_SAMPLES_PER_SIGN}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Data Collection', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                f.close()
                return

    print("\nData collection complete!")
    cap.release()
    cv2.destroyAllWindows()
    f.close()

if __name__ == '__main__':
    collect_data()
