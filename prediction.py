import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']  # Ensure this is the correct key for your model

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Updated labels_dict for alphabets (0-25) and dynamic sequences (26, 27, etc.)
#labels_dict = {0:'B',1:'S'}
labels_dict = {i: chr(65 + i) for i in range(26)}  # A-Z for classes 0-25
labels_dict.update({26: 'hello', 27: 'goodbye', 28: 'thank you', 29: 'welcome', 30: 'please', 31: 'sorry'})  # Adding dynamic sequences

def calculate_distances_and_angles(hand_landmarks):
    data_aux = []

    for i in range(len(hand_landmarks)):
        for j in range(i + 1, len(hand_landmarks)):
            distance = np.sqrt((hand_landmarks[i].x - hand_landmarks[j].x) ** 2 +
                               (hand_landmarks[i].y - hand_landmarks[j].y) ** 2)
            data_aux.append(distance)

    for i in range(0, len(hand_landmarks) - 2, 3):
        angle = np.degrees(np.arctan2(hand_landmarks[i + 2].y - hand_landmarks[i].y,
                                      hand_landmarks[i + 2].x - hand_landmarks[i].x))
        data_aux.append(angle)

    expected_features = 217  # Replace with the actual number of features your model expects
    if len(data_aux) < expected_features:
        data_aux.extend([0] * (expected_features - len(data_aux)))
    elif len(data_aux) > expected_features:
        data_aux = data_aux[:expected_features]

    return data_aux

def process_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for proper color display
    results = hands.process(frame_rgb)

    predicted_text = "No sign detected"
    accuracy = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = [lm for lm in hand_landmarks.landmark]
            data_aux = calculate_distances_and_angles(landmarks)

            probs = model.predict_proba([np.asarray(data_aux)])[0]
            max_proba = max(probs)

            if max_proba >= 0.5:  # Display prediction if above threshold
                predicted_text = labels_dict.get(np.argmax(probs), "Unknown")  # Use the updated labels_dict
                accuracy = max_proba * 100

    return predicted_text, accuracy
