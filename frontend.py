import streamlit as st
import cv2
import prediction as inf  # Import your prediction logic from inference_classifier.py


# Set the Streamlit page configuration
st.set_page_config(page_title="Sign Language Recognition", page_icon="🖐️", layout="wide")


# Title and description
st.title("                                          ")
st.title("Indian Sign Language Prediction 🖐️")
st.markdown("""
Welcome to the Indian Sign Language app!
Press **Start Video Capture** to begin video capture and see the predicted sign language and its accuracy in real-time.
""")


# Custom CSS for button, layout, and text styling
st.markdown("""
    <style>
        /* Modern, sleek design */
        body {
            color: #FFFFFF;
            background-color: #1E1E1E;
        }
        .stApp {
            background: linear-gradient(135deg, #1E1E1E 0%, #2D2D2D 100%);
        }
        .main .block-container {
            padding-top: 2rem;
        }
        h1 {
            color: #FF5733;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        /* Neon effect for buttons */
        .stButton>button {
            background-color: black;
            color: #FF5733;
            border: 2px solid #FF5733;
            border-radius: 20px;
            padding: 10px 25px;
            font-size: 18px;
            font-weight: bold;
            transition: all 0.3s ease;
            box-shadow: 0 0 10px #FF5733;
        }
        .stButton>button:hover {
            background-color: #FF5733;
            color: #1E1E1E;
            box-shadow: 0 0 20px #FF5733;
        }
        /* Futuristic container for predictions */
        .prediction-container {
            background: rgba(30, 30, 30, 0.7);
            border: 2px solid #FF5733;
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0 0 15px rgba(255, 87, 51, 0.5);
        }
        /* Predicted Sign Label */
        .predicted-sign-label {
            font-size: 1.8rem;
            color: white;
            font-weight: bold;
            margin-bottom: 10px;
        }

        /* Predicted Text Styling */
        .predicted-text {
            font-size: 1.5rem;
            color: #ff371d;
            font-weight: bold;
            margin-bottom: 10px;
            padding: 20px;
            border-radius: 8px;  /* Rounded corners */
            border: 1px solid #ff371d;  /* Orange border */
        }

        /* Accuracy Label */
        .accuracy-label {
            font-size: 1.5rem;
            color: white;
            margin-bottom: 10px;
            font-weight: bold;
        }

        /* Accuracy Text Styling */
        .accuracy-text {
            font-size: 1.5rem;
            color: #ff371d;
            font-weight: bold;
            padding: 20px;
            border-radius: 8px;  /* Rounded corners */
            border: 1px solid #ff371d;  /* Orange border */
        }

        /* Center the video frame */
        .video-frame {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: auto; /* Center the frame in the container */
        }

        .video-frame img {
            border-radius: 15px;
            box-shadow: 0 0 20px rgba(255, 87, 51, 0.3);
        }
    </style>
""", unsafe_allow_html=True)


# Create video capture object
cap = cv2.VideoCapture(0)  # Change to 0 for default webcam


# Create a "Start Video Capture" button
run_app = st.button("Start Video Capture")

if run_app:
    # Create three columns for layout: Image, Video, and Prediction
    col1, col2, col3 = st.columns([1, 3, 1])  # Image takes 1/5, Video takes 3/5, Predictions take 1/5

    with col1:
        # Display the image in the first column
        st.image("C:/Users/jayde/Downloads/reference_img.png", caption="Reference Image", use_column_width=True)  # Add your image path here

    with col2:
        # Center-align video frame in the 3 columns
        with st.container():
            st.markdown('<div class="video-frame">', unsafe_allow_html=True)
            frame_holder = st.empty()  # Video frame placeholder
            st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        # Display the predicted sign label and text box
        st.markdown("<div class='predicted-sign-label'>Predicted Sign</div>", unsafe_allow_html=True)
        predicted_text_placeholder = st.empty()

        # Display the accuracy label and text box
        st.markdown("<div class='accuracy-label'>Accuracy (%)</div>", unsafe_allow_html=True)
        accuracy_placeholder = st.empty()

    # Start capturing video
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            st.error("Failed to access the webcam. Make sure it is properly connected.")
            break

        # Use the imported function to get predictions
        predicted_text, accuracy = inf.process_frame(frame)

        # Display the video frame in color and center it
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_holder.image(frame_rgb, channels="RGB", width=720)

        # Display predicted text and accuracy in real-time
        if predicted_text and accuracy >= 50.0:
            predicted_text_placeholder.markdown(f"<div class='predicted-text'>{predicted_text}</div>", unsafe_allow_html=True)
            accuracy_placeholder.markdown(f"<div class='accuracy-text'>{accuracy:.2f}%</div>", unsafe_allow_html=True)
        else:
            predicted_text_placeholder.markdown("<div class='predicted-text'>No sign identified</div>", unsafe_allow_html=True)
            accuracy_placeholder.markdown("<div class='accuracy-text'>N/A</div>", unsafe_allow_html=True)

        # Stop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
