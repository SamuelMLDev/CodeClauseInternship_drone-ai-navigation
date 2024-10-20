# src/main.py

import cv2
from computer_vision import ComputerVision

def main():
    # Initialize the ComputerVision class
    cv = ComputerVision()

    # Capture video from the default webcam (0). 
    # To use a video file instead, replace 0 with the file path, e.g., 'video.mp4'
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("Starting video stream. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Detect faces in the frame
        faces = cv.detect_faces(frame)

        # Draw rectangles around detected faces
        frame_with_faces = cv.draw_faces(frame, faces)

        # Display the resulting frame
        cv2.imshow('Drone AI - Face Detection', frame_with_faces)

        # Exit loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close display windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
