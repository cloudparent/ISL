import os
import cv2
import mediapipe as mp
import numpy as np
from math import sqrt, atan2, degrees
from tqdm import tqdm  # Import tqdm for the progress bar

# Directory to save the data
DATA_DIR = './data'
number_of_classes = 32  # A-Z

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

def calculate_distances_and_angles(hand_landmarks):
    data_aux = []

    # Calculate distances between hand landmarks
    for i in range(len(hand_landmarks)):
        for j in range(i + 1, len(hand_landmarks)):
            distance = sqrt((hand_landmarks[i].x - hand_landmarks[j].x) ** 2 +
                            (hand_landmarks[i].y - hand_landmarks[j].y) ** 2)
            data_aux.append(distance)

    # Calculate angles between points
    for i in range(0, len(hand_landmarks) - 2, 3):
        angle = degrees(atan2(hand_landmarks[i + 2].y - hand_landmarks[i].y,
                              hand_landmarks[i + 2].x - hand_landmarks[i].x))
        data_aux.append(angle)

    return data_aux

def process_images():
    # Iterate through each class
    for j in range(number_of_classes):
        class_dir = os.path.join(DATA_DIR, str(j))
        
        if not os.path.exists(class_dir):
            print(f"Directory for class {j} does not exist, skipping...")
            continue

        images = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]  # Only process .jpg files

        # Initialize tqdm progress bar
        for image in tqdm(images, desc=f'Processing class {j}', unit='image'):
            img_path = os.path.join(class_dir, image)

            try:
                img = cv2.imread(img_path)

                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = hands.process(img_rgb)

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:

                            landmarks = [lm for lm in hand_landmarks.landmark]
                            feature_data = calculate_distances_and_angles(landmarks)

                            # Save feature_data as .npy file
                            np.save(os.path.join(class_dir, f'{os.path.splitext(image)[0]}.npy'), feature_data)
                    else:
                        print(f"No hands detected in {img_path}")

                else:
                    print(f"Failed to read image: {img_path}")

            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

# Call the function to process images
process_images()
