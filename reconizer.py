import cv2
import numpy as np
import onnxruntime as ort
from collections import Counter


class Recognizer:
    def __init__(self):
        self.lpr_session = ort.InferenceSession("weights/lprnet_persian.onnx", providers=['CPUExecutionProvider'])
        self.input_name = self.lpr_session.get_inputs()[0].name
        self.chars = [
        'Alef', 'B', 'P', 'T', 'S', 'Jim', 'Ch', 'H', 'Kh', 'Daal', 
        'Zaal', 'R', 'Z', 'ZH', 'Sin', 'Shin', 'Saad', 'Zaad', 'Taa', 'Zaa', 
        'Ein', 'Ghein', 'F', 'Ghaaf', 'Kaaf', 'Gaaf', 'Laam', 'Mim', 'Noon', 'V', 'He', 'Y',
        'SS', 'DP', 'PT', 'ML',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        '-']
        
    def predict(self,crops):
        input_data = self.preprocess(crops)
        raw_outputs = self.lpr_session.run(None, {self.input_name: input_data})
        logits = raw_outputs[0]  # (Batch_Size, Num_Classes, Time_Steps)
        out = self.postprocess(logits)
        most, count = Counter(out).most_common(1)[0]
        return f'{most} - {count} times'
    
    def postprocess(self,logits):
        decoded_predictions = []
        batch_size = logits.shape[0]
        
        for i in range(batch_size):
            blank_idx = len(self.chars)-1
            pred_matrix = logits[i, :, :]
            
            raw_sequence = np.argmax(pred_matrix, axis=0).tolist()
            compressed_plate = []
            
            pre_c = raw_sequence[0]
            if pre_c != blank_idx:
                compressed_plate.append(self.chars[pre_c])
                
            for c in raw_sequence[1:]:
                if c == pre_c:
                    continue
                    
                if c == blank_idx:
                    pre_c = c
                    continue
                compressed_plate.append(self.chars[c])
                pre_c = c
                
            decoded_predictions.append(''.join(compressed_plate))

        return decoded_predictions
    
    
    def preprocess(self,crops):
        
        if not crops or len(crops) == 0:
            return None
        
        crops = self.deskew_plate(crops)
        batch_input = cv2.dnn.blobFromImages(
            images=crops,
            scalefactor=0.0078125,
            size=(94, 24),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
            crop=False
        )
        
        return batch_input
    
    def deskew_plate(self, crop):
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 50, 150)
            
            contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return crop
                
            c = max(contours, key=cv2.contourArea)
            
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            if len(approx) == 4:
                pts = approx.reshape(4, 2)
                
                rect = np.zeros((4, 2), dtype="float32")
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]
                
                dst = np.array([
                    [0, 0],
                    [94 - 1, 0],
                    [94 - 1, 24 - 1],
                    [0, 24 - 1]
                ], dtype="float32")
                
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(crop, M, (94, 24))
                return warped
                
        except Exception as e:
            pass
            
        return crop