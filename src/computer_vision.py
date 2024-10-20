# src/computer_vision.py

import cv2
import yolov5
import torch
import numpy as np

class ComputerVision:
    def __init__(self, cascade_path='haarcascade_frontalface_default.xml', yolo_model='yolov5s.pt'):
        # Load the pre-trained Haar Cascade classifier for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)
        if self.face_cascade.empty():
            raise IOError("Unable to load the face cascade classifier xml file.")
        
        # Load the YOLOv5 model
        if not torch.cuda.is_available():
            self.device = 'cpu'
        else:
            self.device = 'cuda'
        
        self.yolo = yolov5.load(yolo_model, device=self.device)
        self.yolo.conf = 0.5  # Confidence threshold
        self.yolo.iou = 0.45   # IoU threshold

        # Initialize ORB detector for SLAM
        self.orb = cv2.ORB_create()
        # Initialize BFMatcher for feature matching
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Initialize variables for pose estimation and mapping
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.pose = np.eye(4)  # 4x4 identity matrix for pose
        self.map_points = []

    def detect_faces(self, frame):
        """
        Detects faces in a given frame.

        Args:
            frame (numpy.ndarray): The image frame in which to detect faces.

        Returns:
            list: A list of rectangles where faces were detected.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces

    def draw_faces(self, frame, faces):
        """
        Draws rectangles around detected faces.

        Args:
            frame (numpy.ndarray): The original image frame.
            faces (list): A list of rectangles where faces were detected.

        Returns:
            numpy.ndarray: The image frame with rectangles drawn around faces.
        """
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame

    def detect_objects(self, frame):
        """
        Detects objects in a given frame using YOLOv5.

        Args:
            frame (numpy.ndarray): The image frame in which to detect objects.

        Returns:
            list: A list of detected objects with their bounding boxes and labels.
        """
        results = self.yolo(frame)
        return results.xyxy[0]  # Bounding boxes with coordinates

    def draw_objects(self, frame, objects):
        """
        Draws bounding boxes and labels around detected objects.

        Args:
            frame (numpy.ndarray): The original image frame.
            objects (list): A list of detected objects with bounding box coordinates and labels.

        Returns:
            numpy.ndarray: The image frame with bounding boxes and labels drawn around objects.
        """
        for *box, conf, cls in objects:
            x1, y1, x2, y2 = map(int, box)
            label = self.yolo.names[int(cls)]
            confidence = conf.item()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return frame

    def initialize_slam(self, frame):
        """
        Initializes SLAM by detecting keypoints and descriptors in the first frame.

        Args:
            frame (numpy.ndarray): The initial image frame.

        Returns:
            None
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_keypoints, self.prev_descriptors = self.orb.detectAndCompute(gray, None)

    def update_slam(self, frame):
        """
        Updates SLAM by detecting keypoints, matching with previous frame, and estimating pose.

        Args:
            frame (numpy.ndarray): The current image frame.

        Returns:
            tuple: Updated frame with keypoints and pose information.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        if self.prev_descriptors is not None and descriptors is not None:
            # Match descriptors
            matches = self.bf.match(self.prev_descriptors, descriptors)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Extract location of good matches
            pts_prev = np.float32([self.prev_keypoints[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
            pts_curr = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
            
            # Find essential matrix
            E, mask = cv2.findEssentialMat(pts_curr, pts_prev, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is not None:
                # Recover pose
                _, R, t, mask_pose = cv2.recoverPose(E, pts_curr, pts_prev)
                # Update pose
                self.pose = self.pose @ np.vstack((
                    np.hstack((R, t)),
                    [0, 0, 0, 1]
                ))
            
            # Update map points
            for kp in keypoints:
                self.map_points.append(kp.pt)

            # Draw matches (optional)
            # matched_frame = cv2.drawMatches(frame, keypoints, self.prev_frame, self.prev_keypoints, matches[:10], None, flags=2)
        
        # Update previous keypoints and descriptors
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors

        return frame

    def get_pose(self):
        """
        Returns the current pose of the drone.

        Returns:
            numpy.ndarray: 4x4 transformation matrix representing the pose.
        """
        return self.pose

    def get_map(self):
        """
        Returns the accumulated map points.

        Returns:
            list: List of points in the map.
        """
        return self.map_points
