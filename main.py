from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import hashlib
from typing import List
import uvicorn

from hair_analyzer import analyze_image_color
from cache import init_db, get_cached_result, cache_result

app = FastAPI(title="Color Analyzer API")

# CORS configuration
origins = [
    "http://localhost:3000",
    "http://localhost:5173",  # Vite default port
    "https://hair-color-analyzer-frontend.vercel.app"
    # Add your frontend production URL here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    await init_db()

@app.post("/analyze-color")
async def analyze_color(images: List[UploadFile] = File(...)):
    """
    Analyzes a list of images to determine the dominant color.
    - Caches results based on image content.
    """
    results = []
    for image in images:
        image_bytes = await image.read()
        image_hash = hashlib.sha256(image_bytes).hexdigest()

        # Check cache first
        cached_result = await get_cached_result(image_hash)
        if cached_result:
            rgb, hex_val, closest_color, match_percentage = cached_result
            results.append({
                "filename": image.filename,
                "dominant_color_rgb": rgb,
                "dominant_color_hex": hex_val,
                "closest_color": closest_color,
                "match_percentage": match_percentage,
                "message": f"The dominant color is {hex_val}, which is closest to your predefined color '{closest_color}'.",
                "from_cache": True,
            })
            continue

        # If not in cache, analyze the image
        try:
            analysis_result = analyze_image_color(image_bytes)
            if "error" in analysis_result:
                raise HTTPException(status_code=400, detail=analysis_result["error"])

            # Cache the new result
            await cache_result(
                image_hash,
                analysis_result["dominant_color_rgb"],
                analysis_result["dominant_color_hex"],
                analysis_result["closest_color"],
                analysis_result["match_percentage"]
            )

            results.append({
                "filename": image.filename,
                **analysis_result,
                "from_cache": False,
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred during analysis: {str(e)}")

    return {"results": results}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
