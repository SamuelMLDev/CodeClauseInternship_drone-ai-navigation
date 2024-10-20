# src/main.py

import cv2
from computer_vision import ComputerVision
import numpy as np

def main():
    # Initialize the ComputerVision class with YOLOv5
    cv_module = ComputerVision()

    # Capture video from the default webcam (0).
    # To use a video file instead, replace 0 with the file path, e.g., 'video.mp4'
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("Starting video stream. Press 'q' to exit.")

    slam_initialized = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Detect faces in the frame
        faces = cv_module.detect_faces(frame)

        # Draw rectangles around detected faces
        frame_with_faces = cv_module.draw_faces(frame, faces)

        # Detect objects in the frame using YOLOv5
        objects = cv_module.detect_objects(frame_with_faces)

        # Draw bounding boxes and labels around detected objects
        frame_with_objects = cv_module.draw_objects(frame_with_faces, objects)

        # Initialize SLAM with the first frame
        if not slam_initialized:
            cv_module.initialize_slam(frame_with_objects)
            slam_initialized = True
        else:
            # Update SLAM with the current frame
            frame_with_objects = cv_module.update_slam(frame_with_objects)

        # Retrieve and display the drone's pose
        pose = cv_module.get_pose()
        pose_text = f"Pose:\nRotation Matrix:\n{pose[:3, :3]}\nTranslation Vector:\n{pose[:3, 3]}"
        # Display pose information on the frame (optional)
        # You can format and display the pose as needed

        # Display the resulting frame
        cv2.imshow('Drone AI - Object Detection and SLAM', frame_with_objects)

        # Exit loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close display windows
    cap.release()
    cv2.destroyAllWindows()

    # Optionally, visualize the map points
    map_points = cv_module.get_map()
    if map_points:
        map_image = np.zeros((600, 800, 3), dtype=np.uint8)
        for point in map_points:
            x, y = int(point[0]), int(point[1])
            cv2.circle(map_image, (x, y), 1, (0, 255, 255), -1)
        cv2.imshow('Drone AI - Map', map_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
