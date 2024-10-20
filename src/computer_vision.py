

import cv2

class ComputerVision:
    def __init__(self, cascade_path='haarcascade_frontalface_default.xml'):
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)
        if self.face_cascade.empty():
            raise IOError("Unable to load the face cascade classifier xml file.")

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
