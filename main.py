import cv2
import time
from frame_reader import FrameReader
from utils import motion
# from detector import Detector
# from recognizer import Recognizer

OPERATION = True
def pipeline():
    print("Loading models...")
    reader = FrameReader(source=0).start()
    motion_trigger = motion.MotionDetector(delay=5)
    # detector = Detector(model_path="yolo.onnx")
    # recognizer = Recognizer(model_path="lprnet.onnx")
    
    
    print("Pipeline is running in Standby mode.")
    
    while not reader.stopped:
        ret, frame = reader.get_frame()
        if not ret:
            break
        
        OPERATION = motion_trigger.delayed_check(frame)
        
        cv2.imshow('f',frame)
        if cv2.waitKey(1) == ord('q'):
            break
        
        if not OPERATION:
            print('Standby mode')
            time.sleep(0.01)
            continue
        print('Operating')
            
        

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