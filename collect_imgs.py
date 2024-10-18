import os
import cv2
import time

# Directory to save the data
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 32  # A-Z (26 classes)
dataset_size = 100  # Increase dataset to 200 images per class

cap = cv2.VideoCapture(1)  # Change to 0 to use the default webcam (built-in webcam)

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    # Count the number of images already in the class directory
    current_count = len([name for name in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, name))])

    print(f'Collecting data for class {j}. Currently {current_count} images.')

    # If we already have 200 images, skip this class
    if current_count >= dataset_size:
        print(f'Skipping class {j}, already has {current_count} images.')
        continue

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            time.sleep(2)  # Add a 2-second delay to get ready for image capture
            break

    # Collect more images until we reach 200 for this class
    while current_count < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)  # Display the frame

        # Save the new images in the class directory
        cv2.imwrite(os.path.join(class_dir, f'{current_count}.jpg'), frame)
        current_count += 1

    print(f'Finished collecting data for class {j}.')

cap.release()
cv2.destroyAllWindows()
