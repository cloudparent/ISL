Indian Sign Language Interpreter
This project implements an Indian Sign Language Interpreter using Python, OpenCV, Mediapipe, and Random Forest for classification. It captures images via webcam, processes them to extract hand landmarks, and predicts the corresponding sign language alphabet or dynamic sign.

Description
The project is designed to recognize and predict hand gestures corresponding to the Indian Sign Language alphabets (A-Z) as well as dynamic signs like 'hello', 'goodbye', etc. The system uses:

OpenCV for capturing video from the webcam.
Mediapipe for hand landmark detection.
Random Forest for classifying the gestures based on extracted features.
Features
Dataset Collection: Capture images for each gesture class using a webcam.
Feature Extraction: Calculate distances and angles between hand landmarks.
Model Training: Train a Random Forest classifier using the processed features.
Real-time Prediction: Use a webcam to capture and predict the sign language gestures in real-time.
User Interface: Simple frontend using Streamlit to display the webcam feed, predictions, and accuracy.
How to Use
1. Dataset Creation
Run the create_dataset.py script to collect images for each gesture class. Use the webcam to capture images of your hand in different positions.

2. Feature Processing
Use the process_images.py script to process the captured images and extract features (distances and angles) using Mediapipe.

3. Model Training
Train the Random Forest model by running train_classifier.py. The model is saved as model.p after training.

4. Real-time Prediction
Run the frontend.py script to start the real-time sign language interpreter. The app will display the video feed, predict the sign, and show the prediction accuracy.

Installation
Requirements
Ensure you have Python 3.8 or above installed.
Required Packages
opencv-python
mediapipe
numpy
scikit-learn
tqdm
pickle
streamlit
Running the Project
Clone or download the repository.
Install the required libraries.
Collect images for the dataset using create_dataset.py.
Process the images and extract features using process_images.py.
Train the Random Forest model using train_classifier.py.
Run frontend.py to start the Streamlit app for real-time prediction.
Collaborators
This project was developed by Durvesh Kanade [https://github.com/durvesh00011000100] and Aaryaman Kattali [https://github.com/TheGr8Ak]
