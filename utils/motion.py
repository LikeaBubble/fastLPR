import cv2
import time
import numpy as np

class MotionDetector:
    def __init__(self, threshold=25, min_area=70000, delay=30):
        self.avg_frame = None
        self.threshold = threshold
        self.min_area = min_area
        self.delay = delay
        self.last_motion_time = 0 

    def has_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.avg_frame is None:
            self.avg_frame = gray.astype("float")
            return False

        cv2.accumulateWeighted(gray, self.avg_frame, 0.05)
        base_frame = cv2.convertScaleAbs(self.avg_frame)

        frame_delta = cv2.absdiff(base_frame, gray)
        _, thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)
        
        white_pixels = np.sum(thresh == 255)
        if white_pixels > self.min_area:
            return True
        
        return False
    
    def delayed_check(self, frame):
        movement = self.has_motion(frame)
        current_time = time.time()
        
        if movement:
            self.last_motion_time = current_time
            return True
        
        if current_time - self.last_motion_time < self.delay:
            return True 
            
        return False