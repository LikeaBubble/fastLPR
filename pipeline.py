import threading
import time
from utils.visualizer import Visualizer
import supervision as sv
from datetime import datetime
from utils import storage
from frame_reader import FrameReader
from utils import motion
from detector import Detector
from utils import plate_batcher as pb
from reconizer import Recognizer
from data.database import GateDatabase

class Pipeline:
    def __init__(self):
        self.thread = None
        self.stop_event = threading.Event()
        self.is_running = False
        self.authorizer = True
        self.video_source = './data/samples/2.mp4'
        
        self.current_frame = None
        self.frame_lock = threading.Lock()

    def _loop(self):
        db = GateDatabase()
        reader = FrameReader(source=self.video_source).start()
        motion_trigger = motion.MotionDetector(delay=30)
        det = Detector()
        recognizer = Recognizer()
        tracker = sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30)
        batcher = pb.PlateBatcher(stack_num=10, selected_num=5)
        vz = Visualizer()
        
        frame_idx = 0
        SKIP_FRAMES = 7
        
        print("🚀 Camera Pipeline Started...")
        
        while not self.stop_event.is_set() and not reader.stopped:
            ret, frame = reader.get_frame()
            if not ret:
                break
            frame_idx += 1 
            
            if not motion_trigger.delayed_check(frame):
                time.sleep(0.05)
                continue
            
            if frame_idx % SKIP_FRAMES == 0:
                sv_detections = det.predict(frame)
                if sv_detections:
                    tracked_detections = tracker.update_with_detections(sv_detections)
                    with self.frame_lock:
                        self.current_frame = vz.plot(frame,tracked_detections)
                    crops_status = batcher.update(frame, tracked_detections)
                    
                    if crops_status:
                        crops, entering = crops_status
                        final_plate, plate_score = recognizer.predict(crops)
                        
                        permission = self.log_detected_plate(
                            db=db,
                            plate=final_plate, 
                            confidence=plate_score, 
                            cropped_img=crops[0],
                            is_entering=entering
                        )
                else:
                    with self.frame_lock:
                        self.current_frame = frame
            time.sleep(0.01)

        reader.stop()
        self.is_running = False
        print("🛑 Camera Pipeline Stopped cleanly.")


    def log_detected_plate(self,db:GateDatabase,plate, confidence, cropped_img, is_entering):
        
        img_path = storage.ImageStorageManager.save_crop(cropped_img, plate, is_entering)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        active_session = db.get_active_session(plate)
        owner = db.check_whitelist(plate)
        
        if is_entering:
            if active_session:
                #(Conflict Handling)
                db.insert_traffic_history(
                    owner=active_session['owner_name'],
                    plate=plate,
                    confidence=confidence,
                    entry_time=active_session['entry_time'],
                    exit_time=None,
                    entry_image_path=active_session['entry_image'],
                    exit_image_path=None,
                    duration_minutes='Not exist',
                    details='Warning. Vehicle detected as entring but active session already exist.'
                )
                db.delete_active_session(plate)
            
            db.insert_active_session(plate, current_time, img_path, confidence, owner)
        else:
            if active_session:
                duration = (datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S") - datetime.strptime(active_session['entry_time'], "%Y-%m-%d %H:%M:%S")).total_seconds()/60

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
            else:
                db.insert_traffic_history(
                    owner='Unknown',
                    plate=plate,
                    confidence=confidence,
                    entry_time=current_time,
                    exit_time=current_time,
                    entry_image_path='Not exist',
                    exit_image_path=img_path,
                    duration_minutes='Not exist',
                    details='Warning. Vehicle detected as exiting but no active session found.'
                )
        
        if not self.authorizer:
            return False
        if owner:
            return True
        return False      
    
    def start(self):
        if self.is_running:
            return {"status": "error", "message": "Pipeline is already running"}
        
        self.stop_event.clear()
        self.is_running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return {"status": "success", "message": "Pipeline started"}

    def stop(self):
        if not self.is_running:
            return {"status": "error", "message": "Pipeline is already stopped"}
        
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)
        self.is_running = False
        return {"status": "success", "message": "Pipeline stopped"}
    