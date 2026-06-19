import cv2
import numpy as np 
import supervision as sv
import onnxruntime as ort 


class Detector:
    def __init__(self, conf_threshold=0.5, input_size=640):
        model_path = './weights/det_half.onnx'
        self.conf_threshold = conf_threshold
        self.input_size = input_size

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
        Resize and pad the image while maintaining aspect ratio.
        This is the standard preprocessing for YOLO models.
        """
        h, w = image.shape[:2]
        # Calculate scaling ratio
        scale = min(target_size / h, target_size / w)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create a blank canvas of target size
        canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)

        # Calculate padding offsets to center the image
        dw = (target_size - new_w) // 2
        dh = (target_size - new_h) // 2

        # Place the resized image onto the canvas
        canvas[dh:dh + new_h, dw:dw + new_w] = resized

        # Store padding and scale info for later use (to map boxes back to original image)
        self.pad_info = (dw, dh, scale)

        return canvas
    
    def preprocess(self, image):
        """
        Preprocess the image for YOLO26 inference.
        """
        # Apply letterboxing
        processed_img = self.letterbox(image, self.input_size)

        # Convert BGR to RGB (YOLO models expect RGB)
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

        # Normalize pixel values to [0, 1]
        processed_img = processed_img.astype(np.float32) / 255.0

        # Change data layout from HWC to CHW
        processed_img = np.transpose(processed_img, (2, 0, 1))

        # Add batch dimension (1, C, H, W)
        input_tensor = np.expand_dims(processed_img, axis=0)

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