import cv2
import numpy as np

# Function to classify objects based on width
def classify_object(width):
    # Heuristic to differentiate between a pen and a hand based on width
    if width > 50:  # Adjust this threshold as necessary
        return "Hand"
    else:
        return "Pen"

# Start capturing video from the inbuilt camera
cap = cv2.VideoCapture(0)

# Set frame width and height
cap.set(3, 800)
cap.set(4, 800)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color range for detection (modify these ranges as needed)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Threshold the HSV image to get only skin color
    mask_skin = cv2.inRange(hsv, lower_skin, upper_skin)

    # Find contours
    contours, _ = cv2.findContours(mask_skin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Classify object based on width
        label = classify_object(w)

        # Draw the contour and label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Create a small dialogue box
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()
