from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn
import hashlib

# Import the core analysis function and cache functions
from hair_analyzer import analyze_image_color
from cache import init_db, get_cached_result, cache_result, clear_cache

app = FastAPI(title="Hair Color Analyzer API")

@app.on_event("startup")
async def startup_event():
    """Initializes the database connection on server startup."""
    await init_db()

# Configure CORS (Cross-Origin Resource Sharing)
# This allows your React frontend to communicate with this backend.
origins = [
    "http://localhost:3000",
    "http://localhost:5173", # Default Vite port
    "http://localhost:5174",
    "https://hair-color-analyzer-frontend.vercel.app"
    # Add your deployed frontend URL here
    # "https://your-frontend-domain.com" 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

@app.post("/analyze-hair-color")
async def analyze_hair_color_endpoint(images: List[UploadFile] = File(...)):
    """
    Analyzes one or more uploaded images for hair color.
    It checks for a cached result first before running a full analysis.
    """
    if len(images) > 3:
        raise HTTPException(status_code=400, detail="You can upload a maximum of 3 images.")

    results = []
    for image in images:
        try:
            image_bytes = await image.read()
            # Create a unique hash of the image file
            image_hash = hashlib.sha256(image_bytes).hexdigest()

            # 1. Check cache
            cached_result = await get_cached_result(image_hash)
            if cached_result:
                cached_result['filename'] = image.filename
                cached_result['cached'] = True # Add flag for frontend
                results.append(cached_result)
                continue
            
            # 2. If not in cache, analyze
            analysis_result = analyze_image_color(image_bytes)
            analysis_result['filename'] = image.filename
            
            if "error" not in analysis_result:
                # 3. Cache the new, successful result
                await cache_result(image_hash, analysis_result)
                analysis_result['cached'] = False # Add flag for frontend
            
            results.append(analysis_result)

        except Exception as e:
            results.append({"filename": image.filename, "error": f"An unexpected error occurred: {str(e)}"})
    
    return results

@app.post("/clear-cache")
async def clear_cache_endpoint():
    """
    Endpoint to manually clear the image analysis cache.
    Triggered by the "Clear Cache" button in the frontend.
    """
    try:
        await clear_cache()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

if __name__ == "__main__":
    # This allows you to run the server with `python main.py`
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)