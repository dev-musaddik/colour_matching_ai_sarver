from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List, Optional
import uvicorn
import hashlib
import aiosqlite as aqlite
from pydantic import BaseModel
import os
import uuid
import asyncio
import json as json_lib

# Import business logic
from hair_analyzer import analyze_image_color
from cache import init_db as init_cache_db, get_cached_result, cache_result, clear_cache
from database import initialize_database, DB_FILE
from training_service import process_image_for_training, train_color_model
from analysis_service import load_all_models, analyze_image_with_trained_models

# ----------------------------
# Constants
# ----------------------------
TRAINING_DATA_DIR = "training_data"
MODELS_DIR = "color_models"
MIN_IMAGES_TO_TRAIN = 1  # Allow training with 1-5 images

# Create static directories before mounting
os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

app = FastAPI(title="Hair Color Analyzer API")

# Mount static directory for serving training images
app.mount(f"/{TRAINING_DATA_DIR}", StaticFiles(directory=TRAINING_DATA_DIR), name="training_data")

# ----------------------------
# Pydantic Models
# ----------------------------
class ColorBase(BaseModel):
    name: str
    description: Optional[str] = None

class ColorCreate(ColorBase):
    pass

class Color(ColorBase):
    id: int

    class Config:
        from_attributes = True

class TrainingImage(BaseModel):
    id: int
    image_path: str
    is_processed: bool

    class Config:
        from_attributes = True

# ----------------------------
# Database Connection
# ----------------------------
async def get_db():
    db = await aqlite.connect(DB_FILE)
    db.row_factory = aqlite.Row
    try:
        yield db
    finally:
        await db.close()

@app.on_event("startup")
async def startup_event():
    await init_cache_db()
    await initialize_database()
    await load_all_models()

# ----------------------------
# CORS Configuration
# ----------------------------
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
if ALLOWED_ORIGINS == ["*"]:
    # Development mode - allow all
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    # Production mode - specific origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# ----------------------------
# Health Check Endpoint
# ----------------------------
@app.get("/health")
async def health_check():
    """Health check endpoint for Render and other monitoring services"""
    return {"status": "healthy", "service": "Hair Color Analyzer API"}

# ----------------------------
# Analysis Endpoint
# ----------------------------
@app.post("/analyze")
async def analyze_image_endpoint(images: List[UploadFile] = File(...)):
    """Analyze 1-5 hair color images individually and return separate results with caching"""
    if len(images) < 1:
        raise HTTPException(status_code=400, detail="Please upload at least 1 image.")
    if len(images) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 images allowed.")
    
    # Read all images and check cache for each
    images_data = []
    for image in images:
        img_bytes = await image.read()
        img_hash = hashlib.sha256(img_bytes).hexdigest()
        images_data.append({
            'bytes': img_bytes,
            'hash': img_hash,
            'filename': image.filename
        })
    
    # Check cache for each image and collect results
    results = []
    images_to_analyze = []
    images_to_analyze_indices = []
    
    for idx, img_data in enumerate(images_data):
        cached_result = await get_cached_result(img_data['hash'])
        if cached_result:
            cached_result['cached'] = True
            cached_result['image_index'] = idx + 1
            cached_result['filename'] = img_data['filename']
            results.append(cached_result)
        else:
            images_to_analyze.append(img_data['bytes'])
            images_to_analyze_indices.append(idx)
    
    # Analyze uncached images
    if images_to_analyze:
        analysis_response = await analyze_image_with_trained_models(images_to_analyze)
        
        if "error" in analysis_response:
            raise HTTPException(status_code=400, detail=analysis_response["error"])
        
        # Cache and add new results
        for i, result in enumerate(analysis_response['results']):
            original_idx = images_to_analyze_indices[i]
            result['cached'] = False
            result['image_index'] = original_idx + 1
            result['filename'] = images_data[original_idx]['filename']
            
            # Cache this individual result
            await cache_result(images_data[original_idx]['hash'], result)
            results.append(result)
    
    # Sort results by image_index to maintain upload order
    results.sort(key=lambda x: x['image_index'])
    
    return {
        "num_images": len(results),
        "results": results
    }

# ----------------------------
# Color Management Endpoints
# ----------------------------
@app.post("/colors", response_model=Color, status_code=201)
async def create_color(color: ColorCreate, db: aqlite.Connection = Depends(get_db)):
    try:
        cursor = await db.execute(
            "INSERT INTO colors (name, description) VALUES (?, ?)",
            (color.name, color.description)
        )
        await db.commit()
        color_id = cursor.lastrowid
        return Color(id=color_id, name=color.name, description=color.description)
    except aqlite.IntegrityError:
        raise HTTPException(status_code=400, detail="A color with this name already exists.")

@app.get("/colors", response_model=List[Color])
async def get_all_colors(db: aqlite.Connection = Depends(get_db)):
    cursor = await db.execute("SELECT id, name, description FROM colors ORDER BY created_at DESC")
    colors = await cursor.fetchall()
    return [Color(id=row["id"], name=row["name"], description=row["description"]) for row in colors]

@app.get("/colors/{color_id}", response_model=Color)
async def get_color(color_id: int, db: aqlite.Connection = Depends(get_db)):
    cursor = await db.execute("SELECT id, name, description FROM colors WHERE id = ?", (color_id,))
    color = await cursor.fetchone()
    if color is None:
        raise HTTPException(status_code=404, detail="Color not found.")
    return Color(id=color["id"], name=color["name"], description=color["description"])

@app.delete("/colors/{color_id}", status_code=204)
async def delete_color(color_id: int, db: aqlite.Connection = Depends(get_db)):
    cursor = await db.execute("SELECT image_path FROM training_images WHERE color_id = ?", (color_id,))
    images_to_delete = await cursor.fetchall()
    
    cursor = await db.execute("DELETE FROM colors WHERE id = ?", (color_id,))
    await db.commit()

    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Color not found.")

    for record in images_to_delete:
        try:
            os.remove(record['image_path'])
        except OSError as e:
            print(f"Error deleting file {record['image_path']}: {e}")
    
    color_dir = os.path.join(TRAINING_DATA_DIR, str(color_id))
    if os.path.exists(color_dir) and not os.listdir(color_dir):
        os.rmdir(color_dir)

    return None

# ----------------------------
# Training Data & Model Endpoints
# ----------------------------
@app.post("/colors/{color_id}/images", response_model=TrainingImage)
async def upload_training_image(color_id: int, image: UploadFile = File(...), db: aqlite.Connection = Depends(get_db)):
    await get_color(color_id, db)

    color_dir = os.path.join(TRAINING_DATA_DIR, str(color_id))
    os.makedirs(color_dir, exist_ok=True)

    file_extension = os.path.splitext(image.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(color_dir, unique_filename)

    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await image.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save image: {e}")

    try:
        cursor = await db.execute(
            "INSERT INTO training_images (color_id, image_path) VALUES (?, ?)",
            (color_id, file_path)
        )
        await db.commit()
        image_id = cursor.lastrowid
        return TrainingImage(id=image_id, image_path=file_path, is_processed=False)
    except aqlite.IntegrityError:
        raise HTTPException(status_code=400, detail="Image path already exists.")

@app.get("/colors/{color_id}/images", response_model=List[TrainingImage])
async def get_training_images_for_color(color_id: int, db: aqlite.Connection = Depends(get_db)):
    await get_color(color_id, db)
    cursor = await db.execute(
        "SELECT id, image_path, is_processed FROM training_images WHERE color_id = ? ORDER BY created_at DESC",
        (color_id,)
    )
    images = await cursor.fetchall()
    return [TrainingImage(id=row["id"], image_path=row["image_path"], is_processed=row["is_processed"]) for row in images]

async def process_and_train(color_id: int, progress_queue: asyncio.Queue):
    """Process images and train model with progress updates sent to queue"""
    try:
        async def progress_callback(step, total_steps, message, percentage):
            await progress_queue.put({
                "step": step,
                "total_steps": total_steps,
                "message": message,
                "percentage": percentage,
                "status": "training"
            })
        
        async with aqlite.connect(DB_FILE) as db:
            db.row_factory = aqlite.Row
            
            cursor = await db.execute("SELECT id, image_path FROM training_images WHERE color_id = ? AND is_processed = FALSE", (color_id,))
            images_to_process = await cursor.fetchall()
            
            # Process images with progress
            for idx, image in enumerate(images_to_process):
                await progress_queue.put({
                    "step": idx + 1,
                    "total_steps": len(images_to_process) + 7,  # images + training steps
                    "message": f"Processing image {idx + 1}/{len(images_to_process)}...",
                    "percentage": int(((idx + 1) / (len(images_to_process) + 7)) * 100),
                    "status": "processing"
                })
                await process_image_for_training(image['id'], image['image_path'])
        
        # Train with progress callback
        await train_color_model(color_id, progress_callback)
        await load_all_models()
        
        # Send completion
        await progress_queue.put({
            "step": 100,
            "total_steps": 100,
            "message": "Training complete!",
            "percentage": 100,
            "status": "complete"
        })
    except Exception as e:
        await progress_queue.put({
            "error": str(e),
            "status": "error"
        })

async def training_event_generator(color_id: int):
    """Generate Server-Sent Events for training progress"""
    progress_queue = asyncio.Queue()
    
    # Start training in background
    training_task = asyncio.create_task(process_and_train(color_id, progress_queue))
    
    try:
        while True:
            # Wait for progress update
            progress_data = await asyncio.wait_for(progress_queue.get(), timeout=60.0)
            
            # Send SSE event
            yield f"data: {json_lib.dumps(progress_data)}\n\n"
            
            # Check if complete or error
            if progress_data.get("status") in ["complete", "error"]:
                break
    except asyncio.TimeoutError:
        yield f"data: {json_lib.dumps({'error': 'Training timeout', 'status': 'error'})}\n\n"
    finally:
        if not training_task.done():
            training_task.cancel()

@app.get("/train/{color_id}/stream")
async def trigger_training_stream(color_id: int, db: aqlite.Connection = Depends(get_db)):
    """Stream training progress using Server-Sent Events"""
    await get_color(color_id, db)

    cursor = await db.execute("SELECT COUNT(id) FROM training_images WHERE color_id = ?", (color_id,))
    image_count = (await cursor.fetchone())[0]
    if image_count < MIN_IMAGES_TO_TRAIN:
        raise HTTPException(status_code=400, detail=f"Not enough images to train. Need at least {MIN_IMAGES_TO_TRAIN}, but found {image_count}.")

    return StreamingResponse(
        training_event_generator(color_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# Fallback non-streaming endpoint for compatibility
@app.post("/train/{color_id}", status_code=200)
async def trigger_training_simple(color_id: int, db: aqlite.Connection = Depends(get_db)):
    """Train a color model without streaming (fallback for compatibility)"""
    await get_color(color_id, db)

    cursor = await db.execute("SELECT COUNT(id) FROM training_images WHERE color_id = ?", (color_id,))
    image_count = (await cursor.fetchone())[0]
    if image_count < MIN_IMAGES_TO_TRAIN:
        raise HTTPException(status_code=400, detail=f"Not enough images to train. Need at least {MIN_IMAGES_TO_TRAIN}, but found {image_count}.")

    try:
        # Process images
        async with aqlite.connect(DB_FILE) as db:
            db.row_factory = aqlite.Row
            cursor = await db.execute("SELECT id, image_path FROM training_images WHERE color_id = ? AND is_processed = FALSE", (color_id,))
            images_to_process = await cursor.fetchall()
            for image in images_to_process:
                await process_image_for_training(image['id'], image['image_path'])
        
        # Train model
        await train_color_model(color_id)
        await load_all_models()
        
        return {"message": f"Successfully trained model for color_id {color_id}.", "color_id": color_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

# ----------------------------
# Legacy & Utility Endpoints
# ----------------------------
@app.post("/analyze-legacy")
async def analyze_legacy_endpoint(images: List[UploadFile] = File(...)):
    if len(images) > 1:
        raise HTTPException(status_code=400, detail="The legacy endpoint only supports one image at a time now.")
    
    image = images[0]
    image_bytes = await image.read()
    image_hash = hashlib.sha256(image_bytes).hexdigest()

    cached_result = await get_cached_result(image_hash)
    if cached_result:
        cached_result['filename'] = image.filename
        cached_result['cached'] = True
        return [cached_result]
    
    analysis_result = analyze_image_color(image_bytes)
    analysis_result['filename'] = image.filename
    
    if "error" not in analysis_result:
        await cache_result(image_hash, analysis_result)
        analysis_result['cached'] = False
    
    return [analysis_result]

@app.post("/clear-cache")
async def clear_cache_endpoint():
    try:
        await clear_cache()
        return {"message": "Cache cleared successfully", "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    try:
        async with aqlite.connect("hair_color_cache.db") as db:
            cursor = await db.execute("SELECT COUNT(*) FROM image_cache")
            count = (await cursor.fetchone())[0]
            return {"cached_results": count, "status": "success"}
    except Exception as e:
        return {"cached_results": 0, "error": str(e), "status": "error"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
