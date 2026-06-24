import cv2

class PlateBatcher:
    def __init__(self, stack_num=10, selected_num=5):
        self.stacked = {}
        self.stacked_xyxy = {}
        self.processed_ids = set()
        self.stack_num = stack_num
        self.selected_num = selected_num
    
    def update(self, frame, tracked):
        
        if tracked.tracker_id is None or len(tracked.tracker_id) == 0:
            return None
            
        height, width, _ = frame.shape
        

        for xyxy, confidence, class_id, tracker_id in zip(
            tracked.xyxy,
            tracked.confidence,
            tracked.class_id,
            tracked.tracker_id
        ):
            
            if tracker_id in self.processed_ids:
                continue
                
            if tracker_id not in self.stacked:
                self.stacked[tracker_id] = []
                self.stacked_xyxy[tracker_id] = [] 
                

            x1, y1, x2, y2 = xyxy.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
                
            self.stacked[tracker_id].append([xyxy, confidence, crop])
            self.stacked_xyxy[tracker_id].append(xyxy)
            
            if len(self.stacked[tracker_id]) == self.stack_num:
                is_coming = self.get_entry_status(self.stacked_xyxy[tracker_id])
                best_crops = self.get_best_crops(self.stacked[tracker_id], frame.shape)
                self.processed_ids.add(tracker_id)
                del self.stacked[tracker_id]
                del self.stacked_xyxy[tracker_id]
                return best_crops,is_coming
                
        return None 
    
    def get_entry_status(self,tracked_plate, line_y=400):
        if len(tracked_plate) < 2:
            return None
        
        first_box = tracked_plate[0]
        y_start = (first_box[1] + first_box[3]) / 2
        
        last_box = tracked_plate[-1]
        y_end = (last_box[1] + last_box[3]) / 2
        
        if y_start < line_y and y_end > line_y:
            return True
            
        elif y_start > line_y and y_end < line_y:
            return False
            
        else:
            if y_end > y_start:
                return True
            else:
                return False

    def get_best_crops(self, tracked_plates, img_shape):
        if not tracked_plates:
            return []
            
        height, width, _ = img_shape
        img_center_x = width // 2
        img_center_y = height // 2
        
        scored_crops = []
        
        area_w = 0.1
        center_w = 0.2
        blur_w = 250
        aspect_w = 170
        
        for plate in tracked_plates:
            xyxy, conf, crop = plate
            x1, y1, x2, y2 = xyxy
            
            if crop is None or crop.size == 0:
                continue
                
            area = (x2 - x1) * (y2 - y1)
            
            plate_center_x = (x1 + x2) // 2
            plate_center_y = (y1 + y2) // 2
            center_dis = abs(plate_center_x - img_center_x) + abs(plate_center_y - img_center_y)
            
            blur_score = cv2.Laplacian(crop, cv2.CV_64F).var()
            
            plate_h = y2 - y1
            if plate_h == 0: 
                continue 
            aspect_ratio = (x2 - x1) / plate_h
            aspect_ratio_diff = abs(aspect_ratio - 4.5)
            
            score = (
                (conf * 400) + 
                (area * area_w) - 
                (center_dis * center_w) + 
                (blur_score * blur_w) - 
                (aspect_ratio_diff * aspect_w)
            )
            
            scored_crops.append((score, crop))
        
        if not scored_crops:
            return []
            
        scored_crops.sort(key=lambda x: x[0], reverse=True)
        best_crops = [crop for score, crop in scored_crops[:self.selected_num]]
        
        return best_crops

    def clear_cache(self, active_tracker_ids):
        self.processed_ids = self.processed_ids.intersection(set(active_tracker_ids))
        for t_id in list(self.stacked.keys()):
            if t_id not in active_tracker_ids:
                del self.stacked[t_id]