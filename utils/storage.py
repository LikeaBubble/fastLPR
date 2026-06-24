import os
import cv2
from datetime import datetime

class ImageStorageManager:
    @staticmethod
    def save_crop(plate_crop, vehicle_id: str, is_entering: bool) -> str:
        subdir = 'Enter' if is_entering else 'Exit'
        today_str = datetime.now().strftime("%Y-%m-%d")
        save_dir = f"./data/stored_images/{today_str}/{subdir}"
        os.makedirs(save_dir, exist_ok=True)
        
        time_str = datetime.now().strftime("%H%M%S")
        filename = f"{save_dir}/id_{vehicle_id.replace(' ', '')}_{time_str}.jpg"
        
        success = cv2.imwrite(filename, plate_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        return filename if success else None