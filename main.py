from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn

from hair_analyzer import analyze_image_color

app = FastAPI(title="Hair Color Analyzer API")

# CORS configuration
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5174",  # Vite default port
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

@app.post("/analyze-hair-color")
async def analyze_hair_color(images: List[UploadFile] = File(...)):
    """
    Analyzes up to 3 images to determine the dominant hair colors and find the closest matching hair color set for each.
    """
    if len(images) > 3:
        raise HTTPException(status_code=400, detail="You can upload a maximum of 3 images at a time.")

    results = []
    for image in images:
        try:
            image_bytes = await image.read()
            
            # Analyze the image
            analysis_result = analyze_image_color(image_bytes)
            
            if "error" in analysis_result:
                results.append({
                    "filename": image.filename,
                    "error": analysis_result["error"]
                })
                continue

            results.append({
                "filename": image.filename,
                **analysis_result
            })
        except Exception as e:
            results.append({
                "filename": image.filename,
                "error": f"An error occurred during analysis: {str(e)}"
            })
    
    return results

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)