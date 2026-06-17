import time
import cv2
from frame_reader import FrameReader
# from detector import Detector
# from recognizer import Recognizer

STAND_BY = True
def pipeline():
    # ۱. مقداردهی اولیه ماژول‌ها
    # در این مرحله مدل‌های سنگین بارگذاری می‌شوند
    print("Loading models...")
    reader = FrameReader(source=0).start()
    # detector = Detector(model_path="yolo.onnx")
    # recognizer = Recognizer(model_path="lprnet.onnx")
    
    print("Pipeline is running in Standby mode.")
    
    while not reader.stopped:
        ret, frame = reader.get_frame()
        if not ret:
            break
        
        cv2.imshow('f',frame)
        
        if cv2.waitKey(1)==ord('q'):
            break
        # ۳. بخش استندبای / مانیتورینگ حرکت (اختیاری اما بهینه)
        # movement_detected = check_motion(frame)
        # if not movement_detected:
        #     time.sleep(0.1) # مصرف CPU را به شدت کاهش می‌دهد
        #     continue

        # ۴. ارسال فریم به دتکتور و ترکر
        # detections, plate_batch = detector.process(frame)
        
        # ۵. اگر از یک پلاک مطمئن شد و بچ تصاویر پر شد
        # if len(plate_batch) >= 10:
        #     final_plate_text = recognizer.vote(plate_batch)
        #     save_to_sqlite(final_plate_text)
        #     send_to_fastapi_client(final_plate_text)
            
        time.sleep(0.01)

    reader.stop()

if __name__ == "__main__":
    pipeline()