import cv2
import time
from frame_reader import FrameReader
from utils import motion
from detector import Detector
import supervision as sv 
from utils import plate_batcher as pb
# from recognizer import Recognizer




def pipeline():
    
    frame_idx = 0
    SKIP_FRAMES = 2
    OPERATION = True
    
    
    print("Loading models...")
    reader = FrameReader(source='samples/3.mp4').start()
    motion_trigger = motion.MotionDetector(delay=5)
    det = Detector()
    tracker = sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30)
    batcher = pb.PlateBatcher(stack_num=10,selected_num=5)

    print("Pipeline is running in Standby mode.")
    
    while not reader.stopped:
        st = time.time()
        
        ret, frame = reader.get_frame()
        if not ret:
            break
        frame_idx += 1 
        
        OPERATION = motion_trigger.delayed_check(frame)
        
        cv2.imshow('f',frame)
        if cv2.waitKey(1) == ord('q'):
            break
        
        if not OPERATION:
            print('Standby mode')
            time.sleep(0.01)
            continue
        
        if frame_idx % SKIP_FRAMES == 0 :
            sv_detections = det.predict(frame)
            tracked_detections = tracker.update_with_detections(sv_detections)
            crops = batcher.update(frame,tracked_detections)
            if crops:
                print('*'*10)
                for i,crop in enumerate(crops):
                    cv2.imwrite(f'f{i}.jpg',crop)
        
        nd = time.time()
        print((nd-st)*1000)

        # ۵. اگر از یک پلاک مطمئن شد و بچ تصاویر پر شد
        # if len(plate_batch) >= 10:
        #     final_plate_text = recognizer.vote(plate_batch)
        #     save_to_sqlite(final_plate_text)
        #     send_to_fastapi_client(final_plate_text)
        time.sleep(0.01)

    reader.stop()

if __name__ == "__main__":
    pipeline()