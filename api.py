import os
import asyncio
from fastapi.staticfiles import StaticFiles
from datetime import datetime
from typing import Optional,Union
from pipeline import Pipeline
from pydantic import BaseModel
from data.database import GateDatabase
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

pipeline_manager = Pipeline()
class History(BaseModel):
    owner: Optional[str] = "ناشناس"
    plate: str
    confidence: Optional[float] = None
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_image_path: Optional[str]= None
    exit_image_path: Optional[str]= None
    duration_minutes: Optional[float] = None
    details: Optional[str] = ''
    
class ActiveSessionIn(BaseModel):
    plate: str
    owner: Optional[str] = "ناشناس"
    confidence: Optional[float] = None
    image_path: Optional[str] = None
    
class WhitelistPlate(BaseModel):
    plate: str
    owner_name: Optional[str] = 'ناشناس'

class VideoSourceUpdate(BaseModel):
    source: Union[int, str]
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline_manager.start()
    yield 
    pipeline_manager.stop()

app = FastAPI(title="FastLPR", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("data/stored_images", exist_ok=True)
app.mount("/data/stored_images", StaticFiles(directory="data/stored_images"), name="stored_images")

@app.get("/system/status")
async def get_system_status():
    return {
        "pipeline_active": pipeline_manager.is_running,
        "video_source": pipeline_manager.video_source
    }

@app.post("/system/source")
async def update_video_source(data: VideoSourceUpdate):
    result = pipeline_manager.change_source(data.source)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result
    
@app.post("/system/start")
async def start_camera():
    result = pipeline_manager.start()
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result

@app.post("/system/stop")
async def stop_camera():
    result = pipeline_manager.stop()
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result

@app.get("/whitelist")
async def get_whitelist():
    loop = asyncio.get_running_loop()
    db_data = await loop.run_in_executor(None, lambda: GateDatabase().get_whitelist())
    
    if not db_data:
        return {"status": "success", "data": []}
        
    formatted_data = []
    for row in db_data:
        formatted_data.append({
            "plate": row[1],
            "owner_name": row[2]
        })
        
    return {
        "status": "success",
        "data": formatted_data
    }

@app.post("/whitelist/add")
async def add_to_whitelist(data:WhitelistPlate):
    if len(data.plate) < 5:
        raise HTTPException(status_code=400, detail="فرمت پلاک نامعتبر است")
        
    loop = asyncio.get_running_loop()
    success = await loop.run_in_executor(
        None, 
        lambda: GateDatabase().add_to_whitelist(data.plate, data.owner_name)
    )
    if not success:
        raise HTTPException(status_code=500, detail="خطا در ذخیره دیتابیس")
    return {"status": "success", "message": f"پلاک {data.plate} با موفقیت تایید شد."}

@app.delete("/whitelist/remove/{plate}")
async def remove_from_whitelist(plate: str):
    loop = asyncio.get_running_loop()
    success = await loop.run_in_executor(
        None, 
        lambda: GateDatabase().remove_from_whitelist(plate)
    )
    if not success:
        raise HTTPException(status_code=404, detail="پلاک در لیست سفید یافت نشد")
    return {"status": "success", "message": f"پلاک {plate} از لیست حذف شد."}

@app.delete("/logs/sessions/delete/{plate}")
async def deleteActiveSession(plate: str):
    loop = asyncio.get_running_loop()
    success = await loop.run_in_executor(
        None, 
        lambda: GateDatabase().delete_active_session(plate)
    )
    if not success:
        raise HTTPException(status_code=404, detail="پلاک در لیست سفید یافت نشد")
    return {"status": "success", "message": f"پلاک {plate} از لیست حذف شد."}



@app.get("/logs/today")
async def get_today_logs(limit: int = 20, offset: int = 0):
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, lambda: GateDatabase().get_today_logs(limit=limit,offset=offset))
    if not data:
        return {"status": "خطا", "message":  'هنوز ترددی ثبت نشده است'}
    return {"status": "success", "data": data}

@app.get("/logs")
async def get_recent_logs(limit: int = 20, offset: int = 0):
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, lambda: GateDatabase().get_recent_logs(limit=limit,offset=offset))
    if not data:
        return {"status": "خطا", "message":  'هنوز ترددی ثبت نشده است'}
    return {"status": "success", "data": data}

@app.get("/logs/sessions")
async def get_sessions(limit: int = 20, offset: int = 0):
    loop = asyncio.get_running_loop()
    db_data = await loop.run_in_executor(None, lambda: GateDatabase().get_sessions(limit=limit,offset=offset))
    
    if not db_data:
        return {"status": "success", "data": []}
        
    formatted_data = []
    for row in db_data:
        formatted_data.append({
            "plate": row[1],
            "entry_time": row[2],
            "entry_image": row[3],
            "confidence": row[4],
            "owner_name": row[5]
        })
        
    return {
        "status": "success",
        "data": formatted_data
    }

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        video_frame_generator(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/logs/sessions/add")
async def add_active_session(data: ActiveSessionIn):
    loop = asyncio.get_running_loop()
    success = await loop.run_in_executor(
        None,
        lambda: GateDatabase().insert_active_session(
            data.plate, datetime.now(), data.image_path, data.confidence, data.owner
        ),
    )
    if not success:
        raise HTTPException(status_code=500, detail="خطا در ثبت ورود دستی")
    return {"status": "success", "message": f"ورود پلاک {data.plate} با موفقیت ثبت شد."}

@app.post("/logs/history/add")
async def add_traffic_history(data: History):
    loop = asyncio.get_running_loop()
    success = await loop.run_in_executor(
        None,
        lambda: GateDatabase().insert_traffic_history(
            data.owner,
            data.plate,
            data.confidence,
            data.entry_time ,
            data.exit_time,
            data.entry_image_path,
            data.exit_image_path,
            data.duration_minutes,
            data.details
        ),
    )
    if not success:
        raise HTTPException(status_code=500, detail="خطا در ثبت ورود دستی")
    return {"status": "success", "message": f"پلاک {data.plate} با موفقیت ثبت شد."}

def video_frame_generator():
    while True:
        
        frame = pipeline_manager.current_frame
        
        if frame is None:
            asyncio.sleep(0.1)
            continue
            
        import cv2
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    
