import sqlite3
import logging
from datetime import datetime

logging.basicConfig(
            filename='./data/warnings/gate_system_errors.log', 
            level=logging.ERROR,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

class GateDatabase:
    def __init__(self, db_name="./data/gate_logs.db"):
        self.db_name = db_name
        self._create_tables()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_name, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _create_tables(self):
        with self._get_connection() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS whitelist (
                id INTEGER PRIMARY KEY AUTOINCREMENT, plate_text TEXT UNIQUE NOT NULL, owner_name TEXT)''')
            conn.execute('''CREATE TABLE IF NOT EXISTS active_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, plate_text TEXT NOT NULL, entry_time DATETIME NOT NULL, entry_image TEXT, confidence TEXT, owner_name TEXT)''')
            conn.execute('''CREATE TABLE IF NOT EXISTS traffic_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT, plate_text TEXT NOT NULL, entry_time DATETIME NOT NULL, exit_time DATETIME, duration_minutes REAL, entry_image TEXT, exit_image TEXT, confidence TEXT, owner_name TEXT, details TEXT)''')

    def get_active_session(self, plate: str):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, entry_time, entry_image, owner_name FROM active_sessions WHERE plate_text = ?", (plate,))
                row = cursor.fetchone()
                return {"session_id": row[0], "entry_time": row[1], "entry_image": row[2], "owner_name": row[3]} if row else None
        except sqlite3.Error as e:
            logging.critical(f"خطا در خواندن اطلاعات از دیتابیس{plate}: {e}")
            return False

    def insert_active_session(self, plate, entry_time, image_path, confidence, owner):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO active_sessions (plate_text, entry_time, entry_image, confidence, owner_name) VALUES (?, ?, ?, ?, ?)",
                            (plate, entry_time, image_path, confidence, owner))
            return True if cursor.rowcount>0 else False
        except sqlite3.Error as e:
            logging.critical(f"خطا در دسترسی به دیتابیس{plate}: {e}")
            return False

    def insert_traffic_history(self,
                               owner:str,
                                plate: str,
                                confidence: str,
                                entry_time: datetime = None,
                                exit_time: datetime = None,
                                entry_image_path: str= None,
                                exit_image_path: str = None,
                                duration_minutes: float= None,
                                details: str = None,
                            ):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO traffic_history (owner_name, plate_text, confidence, entry_time, exit_time, entry_image, exit_image, duration_minutes, details) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (owner, plate, confidence, entry_time, exit_time, entry_image_path, exit_image_path, duration_minutes, details))
                return True if cursor.rowcount>0 else False
        except sqlite3.Error as e:
            logging.critical(f"خطا در دسترسی به دیتابیس{plate}: {e}")
            return False
        
    def delete_active_session(self, plate: str):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM active_sessions WHERE plate_text = ?", (plate,))
                return True if cursor.rowcount>0 else False
        except sqlite3.Error as e:
            logging.critical(f"خطا در دسترسی به دیتابیس{plate}: {e}")
            return False
        
    def check_whitelist(self, plate: str):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT owner_name FROM whitelist WHERE plate_text = ?", (plate,))
                row = cursor.fetchone()
                return row[0] if row else None
        except sqlite3.Error as e:
            logging.critical(f"خطا در دسترسی به دیتابیس{plate}: {e}")
            return False
    
    def remove_from_whitelist(self,plate: str):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM whitelist WHERE plate_text = ?", (plate,))
                return True if cursor.rowcount>0 else False
        except sqlite3.Error as e:
            logging.critical(f"خطا در دسترسی به دیتابیس{plate}: {e}")
            return False
        
    def add_to_whitelist(self,plate:str,owner:str):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO whitelist (plate_text,  owner_name) VALUES (?, ?)",
                            (plate, owner))
            return True if cursor.rowcount>0 else False
        except sqlite3.Error as e:
            logging.critical(f"خطا در دسترسی به دیتابیس{plate}: {e}")
            return False
    
    def get_whitelist(self):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM whitelist")
                data = cursor.fetchall()
                return data if data else None
        except sqlite3.Error as e:
            logging.critical(f"خطا در خواندن اطلاعات از دیتابیس: {e}")
            return False
        
    def get_recent_logs(self, limit: int = 20, offset: int = 0):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM traffic_history 
                    ORDER BY entry_time DESC 
                    LIMIT ? OFFSET ?
                """, (limit, offset))
                data = cursor.fetchall()
                return data if data else None
            
        except sqlite3.Error as e:
            logging.critical(f"خطا در خواندن اطلاعات از دیتابیس: {e}")
            return False
    
    def get_today_logs(self, limit: int = 20, offset: int = 0):
        try:
            today_str = datetime.now().strftime("%Y-%m-%d")
            with self._get_connection() as conn:
                cursor = conn.cursor()
                query = """
                    SELECT *
                    FROM traffic_history 
                    WHERE entry_time LIKE ? 
                    ORDER BY entry_time DESC 
                    LIMIT ? OFFSET ?
                """
                cursor.execute(query, (f"{today_str}%", limit, offset))
                data = cursor.fetchall()
                return data if data else None
        
        except sqlite3.Error as e:
            logging.critical(f"خطا در خواندن اطلاعات از دیتابیس: {e}")
            return False
        
    def get_sessions(self, limit: int = 20, offset: int = 0):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM active_sessions 
                    ORDER BY entry_time DESC 
                    LIMIT ? OFFSET ?
                """, (limit, offset))
                data = cursor.fetchall()
                return data if data else None
        except sqlite3.Error as e:
            logging.critical(f"خطا در خواندن اطلاعات از دیتابیس: {e}")
            return False
        
        
        