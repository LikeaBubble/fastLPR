import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        self.NEON_GREEN = (46, 204, 113)
        self.DARK_GRAY = (44, 62, 80)
        self.WHITE = (255, 255, 255)
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX

    def plot(self,frame,tracked_plates,plate_text: str = None):
        out_frame = frame.copy()
        for xyxy, confidence, class_id, tracker_id in zip(
            tracked_plates.xyxy,
            tracked_plates.confidence,
            tracked_plates.class_id,
            tracked_plates.tracker_id
        ):
            out_frame = self.draw_tracking(out_frame,xyxy,tracker_id,plate_text)
        return out_frame
            
    def draw_tracking(self, frame: np.ndarray, bbox: list, track_id: int, plate_text: str = None):

        out_frame = frame.copy()
        
        x1, y1, x2, y2 = map(int, bbox)
        color = self.NEON_GREEN if plate_text else self.DARK_GRAY
        
        length = int((x2 - x1) * 0.15)
        thickness = 2
        
        cv2.line(out_frame, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(out_frame, (x1, y1), (x1, y1 + length), color, thickness)
        cv2.line(out_frame, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(out_frame, (x2, y1), (x2, y1 + length), color, thickness)
        cv2.line(out_frame, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(out_frame, (x1, y2), (x1, y2 - length), color, thickness)
        cv2.line(out_frame, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(out_frame, (x2, y2), (x2, y2 - length), color, thickness)

        label = f"ID: {track_id}"
        if plate_text:
            label += f" | {plate_text}"
            
        font_scale = 0.5
        font_thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(label, self.FONT, font_scale, font_thickness)
        
        bg_x1 = x1
        bg_y1 = y1 - text_h - 10 if y1 - text_h - 10 > 0 else y1
        bg_x2 = x1 + text_w + 10
        bg_y2 = y1 if y1 - text_h - 10 > 0 else y1 + text_h + 10
        
        cv2.rectangle(out_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), self.DARK_GRAY, -1)
        
        text_y = y1 - 5 if y1 - text_h - 10 > 0 else y1 + text_h + 5
        cv2.putText(out_frame, label, (x1 + 5, text_y), self.FONT, font_scale, self.WHITE, font_thickness, cv2.LINE_AA)

        return out_frame