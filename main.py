import time
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI
from frame_reader import FrameReader
from utils import motion
from detector import Detector
import supervision as sv 
from utils import plate_batcher as pb
from reconizer import Recognizer
from database import GateDatabase
import asyncio

IS_RUNNING = True

def camera_pipeline_loop():
    global IS_RUNNING
    
    db = GateDatabase(authorizer=False)
    reader = FrameReader(source='./samples/1.mp4').start()
    motion_trigger = motion.MotionDetector(delay=30)
    det = Detector()
    recognizer = Recognizer()
    tracker = sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30)
    batcher = pb.PlateBatcher(stack_num=10, selected_num=5)
    
    frame_idx = 0
    SKIP_FRAMES = 7
    
    print("🚀 Camera Pipeline Started in Background...")
    print(db.get_todays_logs())
    while IS_RUNNING and not reader.stopped:
        ret, frame = reader.get_frame()
        if not ret:
            break
        frame_idx += 1 
        
        OPERATION = motion_trigger.delayed_check(frame)
        
        if not OPERATION:
            time.sleep(0.05)
            continue
        
        if frame_idx % SKIP_FRAMES == 0:
            sv_detections = det.predict(frame)
            if sv_detections:
                tracked_detections = tracker.update_with_detections(sv_detections)
                crops_status = batcher.update(frame, tracked_detections)
                
                if crops_status:
                    crops,entering = crops_status
                    final_plate, plate_score = recognizer.predict(crops)
                    db.log(
                        plate=final_plate, 
                        confidence=plate_score, 
                        croped_img=crops[0],
                        entering_status=entering
                    )
                    print(f"✅ Plate Logged: {final_plate}")
        
        time.sleep(0.01)

    reader.stop()
    print("🛑 Camera Pipeline Stopped cleanly.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global IS_RUNNING
    IS_RUNNING = True
    
    pipeline_thread = threading.Thread(target=camera_pipeline_loop, daemon=True)
    pipeline_thread.start()
    
    yield 
    
    print("Shutting down API, signaling camera loop to stop...")
    IS_RUNNING = False
    pipeline_thread.join(timeout=3)


app = FastAPI(title="Persian ALPR Edge System", lifespan=lifespan)

@app.get("/logs")
async def get_recent_logs():
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, lambda: GateDatabase().get_todays_logs())
    return {"status": "success", "data": data}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)