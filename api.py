from fastapi import FastAPI, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
from data.database import GateDatabase
import asyncio
from typing import Optional
from pydantic import BaseModel
from pipeline import Pipeline

pipeline_manager = Pipeline()

class WhitelistPlate(BaseModel):
    plate: str
    owner_name: Optional[str] = 'ناشناس'
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline_manager.start()
    yield 
    pipeline_manager.stop()

app = FastAPI(title="FastLPR", lifespan=lifespan)

@app.get("/system/status")
async def get_system_status():
    return {
        "pipeline_active": pipeline_manager.is_running,
        "video_source": pipeline_manager.video_source
    }

    
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
    return {"status": "success", "data": db_data}

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


@app.get("/logs/today")
async def get_recent_logs():
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, lambda: GateDatabase().get_todays_logs())
    return {"status": "success", "data": data}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)