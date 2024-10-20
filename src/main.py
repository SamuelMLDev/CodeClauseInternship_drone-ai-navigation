# src/main.py

import cv2
from computer_vision import ComputerVision
from slam import SLAM  # Assuming SLAM is in a separate module
import numpy as np

def main():
    # Initialize the ComputerVision and SLAM modules
    cv_module = ComputerVision()
    slam_module = SLAM()

    # Capture video from the default webcam (0).
    # To use a video file instead, replace 0 with the file path, e.g., 'video.mp4'
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("Starting video stream with rule-based navigation. Press 'q' to exit.")

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

        # Update SLAM with the current frame
        frame_with_objects = slam_module.update_slam(frame_with_objects)

        # Get the drone's current pose
        pose = slam_module.get_pose()

        # Define target destination (for simplicity, set to origin)
        target = np.array([0.0, 0.0, 0.0])

        # Calculate distance to target
        distance_to_target = np.linalg.norm(pose[:3] - target)

        # Check for obstacles in the forward path
        obstacle_detected = False
        for obj in objects:
            label = obj[5]  # Assuming label is at index 5
            if label in ['person', 'car', 'bicycle']:  # Define obstacle labels
                obstacle_detected = True
                break

        # Decide on action based on rules
        if obstacle_detected:
            # Simple rule: turn right if obstacle detected
            action = 'Turn Right'
        else:
            if distance_to_target > 1.0:
                action = 'Move Forward'
            else:
                action = 'Stop'

        # Execute the action (for simulation, we'll display the action)
        cv2.putText(frame_with_objects, f"Action: {action}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Display the resulting frame
        cv2.imshow('Drone AI - Object Detection, SLAM, and Rule-Based Navigation', frame_with_objects)

        # Optional: Print action to console
        print(f"Action taken: {action}")

        # Exit loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close display windows
    cap.release()
    cv2.destroyAllWindows()

    # Optionally, visualize the map points
    map_points = slam_module.get_map()
    if map_points:
        map_image = np.zeros((600, 800, 3), dtype=np.uint8)
        for point in map_points:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < 800 and 0 <= y < 600:
                cv2.circle(map_image, (x, y), 1, (0, 255, 255), -1)
        cv2.imshow('Drone AI - Map', map_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
