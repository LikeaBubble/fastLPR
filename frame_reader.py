import cv2
from threading import Thread
import time

class FrameReader:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                self.stop()
                break
            
            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                break
                
            self.ret = ret
            self.frame = frame
            
            time.sleep(0.01)

        self.cap.release()

    def get_frame(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True