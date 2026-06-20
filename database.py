import os
import cv2
import sqlite3
import numpy as np 
from datetime import datetime

class GateDatabase:
    def __init__(self, db_name="gate_logs.db"):
        self.db_name = db_name
        self._create_table()
        self.mapping = mapping = {
    'Alef': 'الف',
    'B': 'ب',
    'P': 'پ',
    'T': 'ت',
    'S': 'ث',
    'Jim': 'ج',
    'Ch': 'چ',
    'H': 'ح',
    'Kh': 'خ',
    'Daal': 'د',
    'Zaal': 'ذ',
    'R': 'ر',
    'Z': 'ز',
    'ZH': 'ژ',
    'Sin': 'س',
    'Shin': 'ش',
    'Saad': 'ص',
    'Zaad': 'ض',
    'Taa': 'ط',
    'Zaa': 'ظ',
    'Ein': 'ع',
    'Ghein': 'غ',
    'F': 'ف',
    'Ghaaf': 'ق',
    'Kaaf': 'ک',
    'Gaaf': 'گ',
    'Laam': 'ل',
    'Mim': 'م',
    'Noon': 'ن',
    'V': 'و',
    'He': 'ه',
    'Y': 'ی',

    'SS': 'سیاسی',
    'DP': 'دیپلمات',
    'PT': 'تشریفات',
    'ML': 'معلولین ♿',

    '0': '۰',
    '1': '۱',
    '2': '۲',
    '3': '۳',
    '4': '۴',
    '5': '۵',
    '6': '۶',
    '7': '۷',
    '8': '۸',
    '9': '۹',

    '-': 'خط تیره'}

    def _create_table(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicle_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_text TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                confidence REAL,
                image_path TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def log_vehicle(self, plate, confidence, croped_img):

        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        img_path = self.save_optimized_crop(croped_img,plate)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plate = ''.join([self.mapping[i] for i in plate.split(' ')])
        plate = plate.replace(" ", "")
        
        cursor.execute('''
            INSERT INTO vehicle_logs (plate_text, timestamp, confidence, image_path)
            VALUES (?, ?, ?, ?)
        ''', (plate, current_time, confidence, img_path))
        
        conn.commit()
        conn.close()

    def get_todays_logs(self):
        
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        today = datetime.now().strftime("%Y-%m-%d")
        cursor.execute("SELECT * FROM vehicle_logs WHERE timestamp LIKE ?", (f"{today}%",))
        
        logs = cursor.fetchall()
        conn.close()
        return logs
    
    def save_optimized_crop(self,plate_crop, vehicle_id):
    
        today_str = datetime.now().strftime("%Y-%m-%d")
        save_dir = f"stored_images/{today_str}"
        os.makedirs(save_dir, exist_ok=True)
        
        time_str = datetime.now().strftime("%H%M%S")
        filename = f"{save_dir}/id_{vehicle_id.replace(" ", "")}_{time_str}.jpg"
        
        success = cv2.imwrite(filename,plate_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        
        if success:
            return filename
        return None