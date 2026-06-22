import os
import cv2
import sqlite3
from typing import Optional
from datetime import datetime
from dataclasses import dataclass


@dataclass
class SessionData:
    owner:str
    plate: str
    confidence: str
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    entry_image_path: Optional[str] = None
    exit_image_path: Optional[str] = None
    duration_minutes: Optional[float] = None
    details: Optional[str] = None

@dataclass
class ActiveSessionData:
    owner:str
    plate: str
    confidence: str
    entry_time: Optional[datetime] = None
    entry_image_path: Optional[str] = None
            
class GateDatabase:
    def __init__(self, authorizer = True ,db_name="gate_logs.db"):
        self.authorizer = authorizer
        self.db_name = db_name
        self._create_table()
        self.mapping = {
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
            CREATE TABLE IF NOT EXISTS whitelist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_text TEXT UNIQUE NOT NULL,
                owner_name TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS active_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_text TEXT NOT NULL,
                entry_time DATETIME NOT NULL,
                entry_image TEXT,
                confidence TEXT,
                owner_name TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS traffic_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_text TEXT NOT NULL,
                entry_time DATETIME NOT NULL,
                exit_time DATETIME,
                duration_minutes REAL,
                entry_image TEXT,
                exit_image TEXT,
                confidence TEXT,
                owner_name TEXT,
                details TEXT
            )
        ''')
        
        conn.commit()
        conn.close()

    def log(self, plate, confidence, croped_img, entering_status):

        img_path = self.save_optimized_crop(croped_img,plate,entering_status)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session_details = self.get_active_session(plate)
        owner = self.is_plate_allowed(plate)
                

        if entering_status:
            if session_details:
                self.log_session(
                    SessionData(
                        plate=plate,
                        confidence=confidence,
                        entry_image_path=session_details['entry_image'],
                        entry_time=session_details['entry_time'],
                        exit_time=None,
                        duration_minutes=None,
                        exit_image_path=img_path,
                        owner=session_details['owner_name'],
                        details='Entring status and active session conflict.Active session exist'
                        ))
                self.delete_active_session(plate)
                print('session Commited')
            self.activate_session(
                ActiveSessionData(
                    owner=owner,
                    confidence=confidence,
                    plate=plate,
                    entry_time=current_time,
                    entry_image_path=img_path
                )
            )
            
            
        else:
            session = SessionData(
                        plate=plate,
                        owner=owner,
                        entry_time=current_time,
                        confidence=confidence,
                        exit_time=current_time,
                        exit_image_path=img_path,
                        details='Entring status and active session conflict. No active session found.'
                        )
            if session_details:
                duration = (datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S") - datetime.strptime(session_details['entry_time'], "%Y-%m-%d %H:%M:%S")).total_seconds()/60
                
                session.entry_image_path=session_details['entry_image']
                session.entry_time=session_details['entry_time']
                session.owner=session_details['owner_name']
                session.details='No warning'
                session.duration_minutes = duration
                
            self.log_session(session)
            self.delete_active_session(plate)
            
            
                
        if not self.authorizer:
            return False
        if owner:
            return True
        return False
    
    
    def get_active_session(self, plate):
        conn = sqlite3.connect(self.db_name)
        conn.execute("PRAGMA journal_mode=WAL;")
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, entry_time, entry_image, owner_name FROM active_sessions WHERE plate_text = ?", 
            (plate,)
        )
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "session_id": row[0],
                "entry_time": row[1],
                "entry_image": row[2],
                "owner_name":row[3]
            }
        
        return None
        
    def log_session(self,session:SessionData):
        conn = sqlite3.connect(self.db_name,timeout=10)
        conn.execute("PRAGMA journal_mode=WAL;")
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO traffic_history(plate_text,
                entry_time,
                exit_time,
                duration_minutes,
                entry_image,
                exit_image,
                confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (session.plate,
              session.entry_time,
              session.exit_time,
              session.duration_minutes,
              session.entry_image_path,
              session.exit_image_path,
              session.confidence))
        conn.commit()
        conn.close()
        
    def activate_session(self,session:ActiveSessionData):
        conn = sqlite3.connect(self.db_name,timeout=10)
        conn.execute("PRAGMA journal_mode=WAL;")
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO active_sessions(plate_text,
                entry_time,
                entry_image,
                confidence)
            VALUES (?, ?, ?, ?)
        ''', (session.plate,
              session.entry_time,
              session.entry_image_path,
              session.confidence))
        conn.commit()
        conn.close()

    def get_todays_logs(self):
        
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        today = datetime.now().strftime("%Y-%m-%d")
        cursor.execute("SELECT * FROM active_sessions WHERE entry_time LIKE ?", (f"{today}%",))
        
        logs = cursor.fetchall()
        conn.close()
        return logs
    
    def save_optimized_crop(self,plate_crop, vehicle_id,entring_status):
        subdir = 'Exit'
        if entring_status:
            subdir = 'Enter'
        today_str = datetime.now().strftime("%Y-%m-%d")
        save_dir = f"stored_images/{today_str}/{subdir}"
        os.makedirs(save_dir, exist_ok=True)
        
        time_str = datetime.now().strftime("%H%M%S")
        filename = f"{save_dir}/id_{vehicle_id.replace(' ', '')}_{time_str}.jpg"
        
        success = cv2.imwrite(filename,plate_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        
        if success:
            return filename
        return None
    
    def is_plate_allowed(self, plate):
        conn = sqlite3.connect(self.db_name)
        conn.execute("PRAGMA journal_mode=WAL;")
        cursor = conn.cursor()
        
            
        cursor.execute(
            "SELECT owner_name FROM whitelist WHERE plate_text = ?", 
            (plate,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return result[0]
        else:
            return None 
    
    def get_active_vehicles(self):
        conn = sqlite3.connect(self.db_name)
        conn.execute("PRAGMA journal_mode=WAL;")
        cursor = conn.cursor()
        
        cursor.execute("SELECT plate_text, entry_time FROM active_sessions")
        active_cars = cursor.fetchall()
        
        conn.close()
        return active_cars
    
    def delete_active_session(self, plate):
        conn = sqlite3.connect(self.db_name, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL;")
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "DELETE FROM active_sessions WHERE plate_text = ?", 
                (plate,)
            )
            conn.commit()
            success = cursor.rowcount > 0
        except sqlite3.Error as e:
            print(f"Database Error in delete_active_session: {e}")
            success = False
        finally:
            conn.close()
            
        return success