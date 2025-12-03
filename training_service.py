import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import json
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from hair_analyzer import segment_hair, get_dominant_colors as get_lab_colors
from database import DB_FILE
import aiosqlite as aqlite

# ----------------------------
# Constants
# ----------------------------
MODELS_DIR = "color_models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ----------------------------
# Model & Preprocessing Setup
# ----------------------------
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
resnet.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ----------------------------
# Feature Extraction
# ----------------------------
def get_embedding(image: Image.Image, hair_mask: np.ndarray) -> np.ndarray:
    np_image = np.array(image)
    mask_3d = np.stack([hair_mask]*3, axis=-1)
    masked_image_np = np.where(mask_3d > 0.5, np_image, 0).astype(np.uint8)
    masked_image = Image.fromarray(masked_image_np)

    input_tensor = preprocess(masked_image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        embedding = resnet(input_batch)
    
    return embedding.squeeze().cpu().numpy()

async def process_image_for_training(image_id: int, image_path: str):
    try:
        image = Image.open(image_path).convert("RGB")
        hair_mask = segment_hair(image)
        if hair_mask is None or np.sum(hair_mask) == 0:
            print(f"Skipping image_id {image_id}: No hair detected.")
            return

        lab_colors = get_lab_colors(image, hair_mask, n_colors=5)
        if not lab_colors:
            print(f"Skipping image_id {image_id}: Could not extract LAB colors.")
            return
        
        for color in lab_colors:
            if 'rgb' in color: del color['rgb']
        lab_features_json = json.dumps(lab_colors)

        embedding = get_embedding(image, hair_mask)
        embedding_blob = pickle.dumps(embedding)

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

            if not positive_features:
                raise ValueError("No features found for the target color. Cannot train.")

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
                raise ValueError("Not enough total samples to train a model.")

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
            raise  # Re-raise to be caught by the endpoint

