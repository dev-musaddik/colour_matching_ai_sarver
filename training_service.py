import requests
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
import numpy as np
import cv2
import json
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from hair_analyzer import segment_hair, get_dominant_colors as get_lab_colors, extract_color_signature
from database import DB_FILE
import aiosqlite as aqlite

# ----------------------------
# Constants
# ----------------------------
MODELS_DIR = "color_models"
os.makedirs(MODELS_DIR, exist_ok=True)

# MediaPipe Model for Embeddings
MP_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite"
MP_MODEL_PATH = "mobilenet_v3_small.tflite"

# ----------------------------
# Model & Preprocessing Setup (Lazy Loading)
# ----------------------------
_embedder = None

def get_embedder():
    """Lazy load MediaPipe Image Embedder."""
    global _embedder
    if _embedder is None:
        # Download model if not exists
        if not os.path.exists(MP_MODEL_PATH):
            print(f"Downloading MediaPipe model from {MP_MODEL_URL}...")
            response = requests.get(MP_MODEL_URL)
            with open(MP_MODEL_PATH, "wb") as f:
                f.write(response.content)
            print("Model downloaded.")

        # Use model_asset_buffer to avoid path issues on Windows/Render
        with open(MP_MODEL_PATH, "rb") as f:
            model_content = f.read()
        
        base_options = python.BaseOptions(model_asset_buffer=model_content)
        options = vision.ImageEmbedderOptions(
            base_options=base_options, l2_normalize=True, quantize=False
        )
        _embedder = vision.ImageEmbedder.create_from_options(options)
        print("MediaPipe Image Embedder loaded.")
    return _embedder

# ----------------------------
# Feature Extraction
# ----------------------------
def get_embedding(image: Image.Image, hair_mask: np.ndarray) -> np.ndarray:
    """Extract embedding from hair region using MediaPipe Image Embedder."""
    embedder = get_embedder()
    
    # Apply mask to image (black out background)
    np_image = np.array(image)
    mask_3d = np.stack([hair_mask]*3, axis=-1)
    masked_image_np = np.where(mask_3d > 0.5, np_image, 0).astype(np.uint8)
    
    # Convert to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=masked_image_np)
    
    # Get embedding
    embedding_result = embedder.embed(mp_image)
    
    # Return the first embedding vector as numpy array
    return embedding_result.embeddings[0].embedding

async def process_image_for_training(image_id: int, image_path: str):
    """
    Process a single image for training: segment hair, extract colors, and get embedding.
    Robustly handles errors to prevent pipeline crashes.
    """
    try:
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Skipping image_id {image_id}: Failed to open image. {e}")
            return

        hair_mask = segment_hair(image)
        if hair_mask is None or np.sum(hair_mask) == 0:
            print(f"Skipping image_id {image_id}: No hair detected.")
            return

        # Use extract_color_signature to get both LAB arrays and display dicts
        # We want to save the display_colors (dicts with hex/rgb) because calculate_lab_similarity expects that format
        try:
            lab_colors, display_colors = extract_color_signature(image, hair_mask, n_colors=5)
        except Exception as e:
            print(f"Skipping image_id {image_id}: Error extracting color signature. {e}")
            return
        
        if not display_colors:
            print(f"Skipping image_id {image_id}: Could not extract dominant colors.")
            return
        
        # We save the display_colors (list of dicts) as the JSON features
        # This matches what calculate_lab_similarity expects (list of dicts with 'hex')
        lab_features_json = json.dumps(display_colors)

        try:
            embedding = get_embedding(image, hair_mask)
            embedding_blob = pickle.dumps(embedding)
        except Exception as e:
            print(f"Skipping image_id {image_id}: Error generating embedding. {e}")
            return

        async with aqlite.connect(DB_FILE) as db:
            # Use INSERT OR IGNORE to prevent errors on re-runs
            await db.execute(
                """
                INSERT OR IGNORE INTO image_features (image_id, lab_features, embedding)
                VALUES (?, ?, ?)
                """,
                (image_id, lab_features_json, embedding_blob)
            )
            await db.execute("UPDATE training_images SET is_processed = TRUE WHERE id = ?", (image_id,))
            await db.commit()
            print(f"Successfully processed and saved features for image_id: {image_id}")

    except Exception as e:
        # Catch-all for any other unexpected errors to ensure pipeline doesn't crash
        print(f"Error processing image_id {image_id}: {e}")

# ----------------------------
# Model Training
# ----------------------------
async def train_color_model(color_id: int, progress_callback=None):
    """
    Orchestrates the training of a binary classifier (SVM) for a specific color.
    Now supports training with as few as 1 image and provides progress updates.
    
    Args:
        color_id: ID of the color to train
        progress_callback: Optional async function to call with progress updates
                          Should accept (step, total_steps, message, percentage)
    """
    print(f"Starting training process for color_id: {color_id}")
    
    async def update_progress(step, total_steps, message):
        percentage = int((step / total_steps) * 100)
        print(f"Progress: {percentage}% - {message}")
        if progress_callback:
            await progress_callback(step, total_steps, message, percentage)
    
    total_steps = 7  # Total number of major steps
    current_step = 0
    
    async with aqlite.connect(DB_FILE) as db:
        db.row_factory = aqlite.Row
        try:
            # Step 1: Fetch positive features
            current_step += 1
            await update_progress(current_step, total_steps, "Loading training data...")
            
            cursor = await db.execute("""
                SELECT f.embedding FROM image_features f
                JOIN training_images i ON f.image_id = i.id
                WHERE i.color_id = ?
            """, (color_id,))
            positive_rows = await cursor.fetchall()
            positive_features = [pickle.loads(row['embedding']) for row in positive_rows]
            # Filter out old features (ResNet was 2048, MediaPipe is 1024)
            positive_features = [f for f in positive_features if len(f) == 1024]

            if not positive_features:
                # Instead of crashing, we return early with a message
                print(f"Training aborted for color_id {color_id}: No valid features found (needs 1024 dims).")
                await update_progress(total_steps, total_steps, "Training skipped: No valid images processed.")
                return

            # Step 2: Fetch negative features
            current_step += 1
            await update_progress(current_step, total_steps, "Loading comparison data...")
            
            cursor = await db.execute("""
                SELECT f.embedding FROM image_features f
                JOIN training_images i ON f.image_id = i.id
                WHERE i.color_id != ?
            """, (color_id,))
            negative_rows = await cursor.fetchall()
            negative_features = [pickle.loads(row['embedding']) for row in negative_rows]
            # Filter out old features
            negative_features = [f for f in negative_features if len(f) == 1024]

            if not negative_features:
                current_step += 1
                await update_progress(current_step, total_steps, "Creating embedding-based model...")
                
                print("Warning: No negative samples found. Creating a simple embedding-based model for single-color training.")
                model_path = os.path.join(MODELS_DIR, f"color_{color_id}_embeddings.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump({'embeddings': positive_features, 'type': 'embedding_only'}, f)
                
                performance = json.dumps({"type": "embedding_only", "num_samples": len(positive_features)})
                
                # Delete existing model if any
                await db.execute("DELETE FROM color_models WHERE color_id = ?", (color_id,))
                
                await db.execute(
                    """
                    INSERT INTO color_models (color_id, model_type, model_path, performance_metrics)
                    VALUES (?, 'embedding', ?, ?)
                    """,
                    (color_id, model_path, performance)
                )
                await db.commit()
                
                await update_progress(total_steps, total_steps, "Training complete!")
                print(f"Successfully saved embedding-only model for color_id: {color_id}")
                return

            # Step 3: Prepare training data
            current_step += 1
            await update_progress(current_step, total_steps, "Preparing training dataset...")
            
            X = np.array(positive_features + negative_features)
            y = np.array([1] * len(positive_features) + [0] * len(negative_features))

            if len(X) < 2:
                 # Should be covered by checks above, but just in case
                print(f"Training aborted for color_id {color_id}: Not enough total samples.")
                await update_progress(total_steps, total_steps, "Training skipped: Not enough data.")
                return

            # Use stratified split only if we have at least 2 samples per class
            if len(positive_features) >= 2 and len(negative_features) >= 2:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            else:
                # For very small datasets, use all data for training
                X_train, y_train = X, y
                X_test, y_test = X, y  # Use same data for evaluation (not ideal but necessary)

            # Step 4: Train SVM classifier
            current_step += 1
            await update_progress(current_step, total_steps, "Training AI model...")
            
            model = SVC(kernel='linear', probability=True, random_state=42)
            model.fit(X_train, y_train)

            # Step 5: Evaluate model
            current_step += 1
            await update_progress(current_step, total_steps, "Evaluating model accuracy...")
            
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            performance = json.dumps({"accuracy": acc})
            print(f"Trained model for color_id {color_id} with accuracy: {acc:.2f}")

            # Step 6: Save model file
            current_step += 1
            await update_progress(current_step, total_steps, "Saving model...")
            
            model_path = os.path.join(MODELS_DIR, f"color_{color_id}_svc.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            # Step 7: Update database
            current_step += 1
            await update_progress(current_step, total_steps, "Finalizing...")
            
            # Delete existing model entry if any to avoid UNIQUE constraint error
            await db.execute("DELETE FROM color_models WHERE color_id = ?", (color_id,))
            
            await db.execute(
                """
                INSERT INTO color_models (color_id, model_type, model_path, performance_metrics)
                VALUES (?, 'svm', ?, ?)
                """,
                (color_id, model_path, performance)
            )
            await db.commit()
            
            await update_progress(total_steps, total_steps, "Training complete!")
            print(f"Successfully saved model and updated database for color_id: {color_id}")

        except Exception as e:
            print(f"Training failed for color_id {color_id}: {e}")
            # We don't raise here to avoid crashing the endpoint, but we log the error
            await update_progress(total_steps, total_steps, f"Training failed: {str(e)}")

