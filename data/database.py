import sqlite3
from datetime import datetime


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
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, entry_time, entry_image, owner_name FROM active_sessions WHERE plate_text = ?", (plate,))
            row = cursor.fetchone()
            return {"session_id": row[0], "entry_time": row[1], "entry_image": row[2], "owner_name": row[3]} if row else None

    def insert_active_session(self, plate, entry_time, image_path, confidence, owner):
        with self._get_connection() as conn:
            conn.execute("INSERT INTO active_sessions (plate_text, entry_time, entry_image, confidence, owner_name) VALUES (?, ?, ?, ?, ?)",
                         (plate, entry_time, image_path, confidence, owner))

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
        with self._get_connection() as conn:
            conn.execute("INSERT INTO traffic_history (owner_name, plate_text, confidence, entry_time, exit_time, entry_image, exit_image, duration_minutes, details) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                         (owner, plate, confidence, entry_time, exit_time, entry_image_path, exit_image_path, duration_minutes, details))

    def delete_active_session(self, plate: str):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM active_sessions WHERE plate_text = ?", (plate,))

    def check_whitelist(self, plate: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT owner_name FROM whitelist WHERE plate_text = ?", (plate,))
            row = cursor.fetchone()
            return row[0] if row else None