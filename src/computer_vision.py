# src/computer_vision.py

import cv2
import yolov5
import torch

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
