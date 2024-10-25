# src/main.py

import cv2
from computer_vision import ComputerVision
from slam import SLAM  
import numpy as np

def main():
    
    cv_module = ComputerVision()
    slam_module = SLAM()

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

        
        faces = cv_module.detect_faces(frame)

        
        frame_with_faces = cv_module.draw_faces(frame, faces)

        
        objects = cv_module.detect_objects(frame_with_faces)

        
        frame_with_objects = cv_module.draw_objects(frame_with_faces, objects)

        
        frame_with_objects = slam_module.update_slam(frame_with_objects)

        
        pose = slam_module.get_pose()

        
        target = np.array([0.0, 0.0, 0.0])

        
        distance_to_target = np.linalg.norm(pose[:3] - target)

        
        obstacle_detected = False
        for obj in objects:
            label = obj[5] 
            if label in ['person', 'car', 'bicycle']:  
                obstacle_detected = True
                break

        
        if obstacle_detected:
            
            action = 'Turn Right'
        else:
            if distance_to_target > 1.0:
                action = 'Move Forward'
            else:
                action = 'Stop'

        
        cv2.putText(frame_with_objects, f"Action: {action}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        
        cv2.imshow('Drone AI - Object Detection, SLAM, and Rule-Based Navigation', frame_with_objects)

        
        print(f"Action taken: {action}")

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    cap.release()
    cv2.destroyAllWindows()

    
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
