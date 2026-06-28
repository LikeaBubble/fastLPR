import os
import cv2
import time
import threading
import supervision as sv
from utils import motion
from utils import storage
from datetime import datetime
from detector import Detector
from recognizer import Recognizer
from frame_reader import FrameReader
from utils import plate_batcher as pb
from data.database import GateDatabase
from utils.visualizer import Visualizer


class Pipeline:
    def __init__(self):
        self.display_thread = None
        self.processing_thread = None
        self.stop_event = threading.Event()
        self.is_running = False
        self.authorizer = False
        self.video_source = './data/samples/5.mp4'
        self.reader = None

        self.current_frame = None
        self._latest_raw_frame = None
        self.frame_lock = threading.Lock()

        self._last_detection = None
        self.DETECTION_OVERLAY_HOLD_SECONDS = 1.0


    def _display_loop(self):
        vz = Visualizer()
        print("🖼️  Display thread started...")

        while not self.stop_event.is_set() and not self.reader.stopped:
            ret, frame = self.reader.get_frame()
            if not ret:
                break

            now = time.time()
            with self.frame_lock:
                self._latest_raw_frame = frame

                hold = self._last_detection
                if hold and now < hold["until"]:
                    display_frame = vz.plot(frame, hold["detections"])
                else:
                    display_frame = vz.draw_trigger_line(frame)

                self.current_frame = display_frame

            time.sleep(0.005)

        print("🖼️  Display thread stopped.")


    def _processing_loop(self):
        db = GateDatabase()
        motion_trigger = motion.MotionDetector(delay=30)
        det = Detector()
        recognizer = Recognizer()
        tracker = sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30)
        batcher = pb.PlateBatcher(stack_num=10, selected_num=5)

        frame_idx = 0
        SKIP_FRAMES = 7

        print("🚀 Processing thread started...")

        while not self.stop_event.is_set() and not self.reader.stopped:
            with self.frame_lock:
                frame = self._latest_raw_frame.copy() if self._latest_raw_frame is not None else None

            if frame is None:
                time.sleep(0.02)
                continue

            frame_idx += 1

            if not motion_trigger.delayed_check(frame):
                time.sleep(0.05)
                continue

            if frame_idx % SKIP_FRAMES == 0:
                sv_detections = det.predict(frame)
                tracked_detections = tracker.update_with_detections(sv_detections)

                if len(tracked_detections) > 0:
                    with self.frame_lock:
                        self._last_detection = {
                            "detections": tracked_detections,
                            "until": time.time() + self.DETECTION_OVERLAY_HOLD_SECONDS,
                        }

                    crops_status = batcher.update(frame, tracked_detections)

                    if crops_status:
                        crops, entering = crops_status
                        final_plate, plate_score = recognizer.predict(crops)

                        self.log_detected_plate(
                            db=db,
                            plate=final_plate,
                            confidence=plate_score,
                            cropped_img=crops[0],
                            is_entering=entering
                        )

            time.sleep(0.01)

        print("🛑 Processing thread stopped.")

    def log_detected_plate(self, db: GateDatabase, plate, confidence, cropped_img, is_entering):
        owner = db.check_whitelist(plate)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        active_session = db.get_active_session(plate)

        if self.authorizer and not owner:
                db.insert_traffic_history(
                    owner='Unknown (Non-Whitelisted)',
                    plate=plate,
                    confidence=confidence,
                    entry_time=current_time,
                    exit_time=current_time,
                    entry_image_path=storage.ImageStorageManager.save_crop(cropped_img, plate, is_entering),
                    exit_image_path=None,
                    duration_minutes=0,
                    details='Permission Denied - Vehicle Not in Whitelist'
                )
                return False

        img_path = storage.ImageStorageManager.save_crop(cropped_img, plate, is_entering)

        if is_entering:
            if active_session:
                db.insert_traffic_history(
                    owner=active_session['owner_name'],
                    plate=plate,
                    confidence=confidence,
                    entry_time=active_session['entry_time'],
                    exit_time=None,
                    entry_image_path=active_session['entry_image'],
                    exit_image_path=None,
                    duration_minutes='Not exist',
                    details='Warning. Vehicle detected as entering but active session already exist.'
                )
                db.delete_active_session(plate)

            db.insert_active_session(plate, current_time, img_path, confidence, owner)
            return True if self.authorizer else False

        else:
            if active_session:
                entry_time_clean = active_session['entry_time'].split('.')[0]
                duration = (datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S") - datetime.strptime(entry_time_clean, "%Y-%m-%d %H:%M:%S")).total_seconds() / 60

                db.insert_traffic_history(
                    owner=active_session['owner_name'],
                    plate=plate,
                    confidence=confidence,
                    entry_time=active_session['entry_time'],
                    exit_time=current_time,
                    entry_image_path=active_session['entry_image'],
                    exit_image_path=img_path,
                    duration_minutes=duration,
                    details='No Warning'
                )
                db.delete_active_session(plate)
                return True if self.authorizer else False
            else:
                db.insert_traffic_history(
                    owner=owner,
                    plate=plate,
                    confidence=confidence,
                    entry_time=current_time,
                    exit_time=current_time,
                    entry_image_path='Not exist',
                    exit_image_path=img_path,
                    duration_minutes='Not exist',
                    details='Warning. Whitelisted vehicle detected as exiting but no active session found.'
                )
                return True if self.authorizer else False

    def start(self):
        if self.is_running:
            return {"status": "error", "message": "Pipeline is already running"}

        self.stop_event.clear()
        self.is_running = True
        self.current_frame = None
        self._latest_raw_frame = None
        self._last_detection = None

        self.reader = FrameReader(source=self.video_source).start()
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.display_thread.start()
        self.processing_thread.start()

        return {"status": "success", "message": "Pipeline started"}

    def stop(self):
        if not self.is_running:
            return {"status": "error", "message": "Pipeline is already stopped"}

        self.stop_event.set()
        if self.display_thread:
            self.display_thread.join(timeout=3)
        if self.processing_thread:
            self.processing_thread.join(timeout=3)
        if self.reader:
            self.reader.stop()
        self.is_running = False
        return {"status": "success", "message": "Pipeline stopped"}

    def change_source(self, new_source):
        parsed_source = None
        source_type = "unknown"

        if isinstance(new_source, int) or (isinstance(new_source, str) and new_source.isdigit()):
            parsed_source = int(new_source)
            source_type = "Local Camera"

        elif isinstance(new_source, str) and (new_source.startswith("rtsp://") or new_source.startswith("http://")):
            parsed_source = new_source
            source_type = "IP Camera"

        elif isinstance(new_source, str):
            if os.path.exists(new_source):
                parsed_source = new_source
                source_type = "Video File"
            else:
                return {"status": "error", "message": f"فایل ویدیویی در مسیر یافت نشد: {new_source}"}
        else:
            return {"status": "error", "message": "فرمت سورس تصویر نامعتبر است"}

        self.video_source = parsed_source

        if self.is_running:
            self.stop()
            start_result = self.start()

            if start_result.get("status") == "error":
                return {
                    "status": "error",
                    "message": f"سورس تغییر کرد اما اجرای آن با خطا مواجه شد: {start_result['message']}"
                }

        return {
            "status": "success",
            "message": f"منبع تصویر با موفقیت به {source_type} تغییر یافت",
            "source_type": source_type,
            "source": parsed_source
        }