from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List

# Import the new core functions
from hair_analyzer import analyze_hair_color_from_image, find_closest_match
from database import init_db, add_trained_color, get_all_trained_colors, clear_all_data

app = FastAPI(title="Hair Color Trainer & Analyzer API")

@app.on_event("startup")
async def startup_event():
    """Initializes the database on server startup."""
    await init_db()

# Configure CORS to allow the frontend to communicate with this backend
origins = [
   "http://localhost:3000",
    "http://localhost:5173", # Default Vite port
    "http://localhost:5174",
    "https://hair-color-analyzer-frontend.vercel.app"
    # Add your deployed frontend URL here for production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/train")
async def train_new_color(
    color_name: str = Form(...),
    images: List[UploadFile] = File(...)
):
    """
    Admin endpoint to upload one or more training images, assign them a name,
    extract their color signatures, and save them to the database.
    """
    if not color_name:
        raise HTTPException(status_code=400, detail="Color name is required.")
    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required.")

    saved_colors = []
    for image in images:
        image_bytes = await image.read()
        
        analysis = analyze_hair_color_from_image(image_bytes)
        if "error" in analysis:
            # Skip this image or raise an error. For now, we'll skip.
            print(f"Could not analyze {image.filename}: {analysis['error']}")
            continue

        lab_signature = analysis["lab_signature"]
        
        try:
            await add_trained_color(color_name, lab_signature, image.filename)
            saved_colors.append(analysis["display_colors"])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database error for {image.filename}: {str(e)}")

    if not saved_colors:
        raise HTTPException(status_code=400, detail="None of the provided images could be processed.")

    return {
        "message": f"Successfully trained color '{color_name}' with {len(saved_colors)} image(s).",
        "trained_color_name": color_name,
        "saved_signatures_count": len(saved_colors)
    }


@app.post("/analyze")
async def analyze_user_image(images: List[UploadFile] = File(...)):
    """
    User endpoint to upload one or more images, extract their color signatures,
    and find the closest match for each from the trained dataset.
    """
    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required.")

    # Get all trained colors once
    trained_colors = await get_all_trained_colors()
    if not trained_colors:
        raise HTTPException(status_code=404, detail="No colors have been trained in the system yet.")

    results = []
    for image in images:
        image_bytes = await image.read()

        # 1. Analyze the user's image
        user_analysis = analyze_hair_color_from_image(image_bytes)
        if "error" in user_analysis:
            results.append({
                "filename": image.filename,
                "error": user_analysis["error"]
            })
            continue

        user_lab_signature = user_analysis["lab_signature"]

        # 2. Find the best match
        best_match = find_closest_match(user_lab_signature, trained_colors)
        if not best_match:
            results.append({
                "filename": image.filename,
                "error": "Could not find a suitable match."
            })
            continue
        
        results.append({
            "filename": image.filename,
            "user_hair_colors": user_analysis["display_colors"],
            "closest_match": best_match
        })

    return {"analysis_results": results}

@app.get("/trained-colors")
async def get_trained_colors():
    """Endpoint to retrieve all currently trained colors."""
    colors = await get_all_trained_colors()
    return {"trained_colors": colors}


@app.post("/admin/clear-database")
async def clear_database():
    """A simple admin endpoint to clear all data for a fresh start."""
    try:
        await clear_all_data()
        return {"message": "All trained color data has been cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear database: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)