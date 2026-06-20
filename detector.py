import cv2
import numpy as np 
import supervision as sv
import onnxruntime as ort 


class Detector:
    def __init__(self, conf_threshold=0.5, input_size=640):
        model_path = './weights/det_half.onnx'
        self.conf_threshold = conf_threshold
        self.input_size = input_size
        self.pad_info = (0, 0, 1.0)
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        
    def predict(self, image):
        input_tensor = self.preprocess(image)
        
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )
        
        boxes, confidences, class_ids = self.postprocess(outputs, image.shape)
        
        sv_detections = sv.Detections(
            xyxy=boxes,
            confidence=confidences,
            class_id=class_ids
        )
        
        return sv_detections
    
    def letterbox(self, image, target_size):
        """
        Fast C++ backed letterboxing using cv2.copyMakeBorder.
        """
        h, w = image.shape[:2]
        
        # Calculate scaling ratio
        scale = min(target_size / h, target_size / w)
        new_w, new_h = int(w * scale), int(h * scale)

        # 1. Fast Resize
        if scale != 1.0:
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = image

        # 2. Calculate padding offsets
        top = (target_size - new_h) // 2
        bottom = target_size - new_h - top
        left = (target_size - new_w) // 2
        right = target_size - new_w - left

        # 3. Fast Padding using C++ backend (replaces np.full and array slicing)
        padded_img = cv2.copyMakeBorder(
            resized, 
            top, bottom, left, right, 
            cv2.BORDER_CONSTANT, 
            value=(114, 114, 114)
        )
        self.pad_info = (left, top, scale)

        return padded_img
    
    def preprocess(self, image):
        """
        Preprocess the image for YOLO26 inference.
        """
        processed_img = self.letterbox(image, self.input_size)
        
        input_tensor = cv2.dnn.blobFromImage(
            image=processed_img,
            scalefactor=1/255.0,            # Normalize pixels to [0, 1]
            size=(self.input_size, self.input_size), 
            mean=(0, 0, 0),                 # No mean subtraction required for YOLO
            swapRB=True,                    # Convert BGR to RGB instantly
            crop=False                      # We already letterboxed, no cropping needed
        )

        return input_tensor
    
    def postprocess(self, outputs, original_shape):
        """
        Post-process the YOLO output and return raw NumPy arrays for Tracker.
        """
        detections = outputs[0][0]  
        confidences = detections[:, 4]  
        mask = confidences > self.conf_threshold  
        filtered_detections = detections[mask]  
        
        if len(filtered_detections) == 0:
            return np.empty((0, 4), dtype=np.int32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)
        
        boxes = filtered_detections[:, :4]  
        confidences = filtered_detections[:, 4]  
        class_ids = filtered_detections[:, 5].astype(np.int32)  
        
        dw, dh, scale = self.pad_info
        orig_h, orig_w = original_shape[:2]
        
        boxes[:, 0] = (boxes[:, 0] - dw) / scale
        boxes[:, 1] = (boxes[:, 1] - dh) / scale
        boxes[:, 2] = (boxes[:, 2] - dw) / scale
        boxes[:, 3] = (boxes[:, 3] - dh) / scale
        
        boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)
        
        boxes = boxes.astype(np.int32)
        
        return boxes, confidences, class_ids