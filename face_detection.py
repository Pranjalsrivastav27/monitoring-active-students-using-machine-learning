import cv2
import numpy as np
import time

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize variables
active_time_threshold = 10  # Minimum time (in seconds) for a face to be considered active
active_times = {}  # Dictionary to store active times for each face
start_time = {}     # Dictionary to store start times for each face

# Start capturing video from the default camera (0)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around the detected faces and track active times
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, 'FACE', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Check if face is already being tracked
        if (x, y, w, h) not in active_times:
            active_times[(x, y, w, h)] = 0
            start_time[(x, y, w, h)] = time.time()
        else:
            active_times[(x, y, w, h)] += time.time() - start_time[(x, y, w, h)]
            start_time[(x, y, w, h)] = time.time()

        # Print active time for each face on the frame
        text = "Face at ({}, {}) appeared for {:.2f} seconds.".format(x, y, active_times[(x, y, w, h)])
        cv2.putText(frame, text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if active_times[(x, y, w, h)] >= active_time_threshold:
            print("Student at ({}, {}) is active for {:.2f} seconds.".format(x, y, active_times[(x, y, w, h)]))
            active_times[(x, y, w, h)] = 0

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()