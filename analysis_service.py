import pickle
import numpy as np
import json
from PIL import Image
import os
import io

from database import DB_FILE
import aiosqlite as aqlite
from training_service import get_embedding, get_lab_colors
from hair_analyzer import segment_hair, color_difference_lab, estimate_image_properties, extract_color_signature
from skimage.color import rgb2lab

# In-memory cache for loaded models to avoid constant file I/O
loaded_models = {}

async def load_all_models():
    """
    Loads all models from the database into the in-memory cache.
    This should be called at startup and after training.
    """
    print("Loading all trained models...")
    async with aqlite.connect(DB_FILE) as db:
        db.row_factory = aqlite.Row
        cursor = await db.execute("""
            SELECT cm.id, cm.model_path, cm.color_id, c.name 
            FROM color_models cm
            JOIN colors c ON cm.color_id = c.id
        """)
        models_to_load = await cursor.fetchall()

        for model_record in models_to_load:
            model_path = model_record['model_path']
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                    loaded_models[model_record['color_id']] = {
                        "model": model,
                        "name": model_record['name'],
                        "id": model_record['color_id']
                    }
        print(f"Loaded {len(loaded_models)} models.")

def calculate_lab_similarity(detected_lab_colors, reference_lab_features_json_list):
    """
    Calculates a similarity score based on LAB color differences (DeltaE).
    Lower score is better.
    
    Args:
        detected_lab_colors: List of numpy arrays representing LAB values of the input image.
        reference_lab_features_json_list: List of JSON strings, each containing a list of reference colors (dicts with 'hex').
    """
    total_score = 0
    
    for d_lab in detected_lab_colors:
        # d_lab is already a numpy array [L, a, b]
        
        best_image_score = float('inf')
        # Compare against each set of reference features (from each training image)
        for ref_json in reference_lab_features_json_list:
            ref_color_dicts = json.loads(ref_json)
            
            # Find the closest color in this specific reference image
            min_diff = float('inf')
            for ref_color in ref_color_dicts:
                # Convert reference Hex -> RGB -> LAB
                # ref_color['hex'] is like "#RRGGBB"
                hex_str = ref_color['hex'].lstrip('#')
                rgb_tuple = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
                
                # Normalize RGB to 0-1 range for rgb2lab
                rgb_norm = np.array(rgb_tuple).reshape(1, 1, 3) / 255.0
                ref_lab = rgb2lab(rgb_norm).reshape(3)
                
                diff = color_difference_lab(d_lab, ref_lab)
                if diff < min_diff:
                    min_diff = diff
            
            if min_diff < best_image_score:
                best_image_score = min_diff
        
        # Weight by the percentage of this detected color? 
        # The detected_lab_colors list doesn't have percentage info directly associated in this loop if we iterate just arrays.
        # However, the original code tried to access ['percentage'] from the array which failed.
        # We should probably pass weights or just average. 
        # For now, let's assume equal weight or simpler: just sum of best matches.
        # But wait, `detected_lab_colors` from `extract_color_signature` is just a list of arrays.
        # The percentages are in `display_colors`.
        # Let's just average the score for now to keep it simple and robust.
        total_score += best_image_score
        
    return total_score / len(detected_lab_colors) if detected_lab_colors else float('inf')


async def analyze_single_image(image_bytes: bytes):
    """
    Analyzes a single image using all loaded custom models.
    Returns analysis result for one image.
    """
    if not loaded_models:
        return {"error": "No trained models are available for analysis."}

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        hair_mask = segment_hair(image)
        if hair_mask is None or np.sum(hair_mask) < (hair_mask.size * 0.01):
            return {"error": "No hair detected in the image."}

        # 1. Extract features from the input image
        embedding = get_embedding(image, hair_mask)
        
        # Use extract_color_signature to get both LAB arrays and display dicts
        detected_lab_colors, display_colors = extract_color_signature(image, hair_mask, n_colors=3)

        if not detected_lab_colors:
            return {"error": "Could not extract dominant colors from hair."}

        # 2. Run inference with all loaded models
        svm_results = []
        print(f"DEBUG: Analyzing image with {len(loaded_models)} models.")
        for color_id, model_data in loaded_models.items():
            model = model_data['model']
            
            # Handle both SVM and embedding-only models
            if isinstance(model, dict) and model.get('type') == 'embedding_only':
                # For embedding-only models, calculate similarity
                ref_embeddings = model['embeddings']
                similarities = [np.dot(embedding, ref_emb) / (np.linalg.norm(embedding) * np.linalg.norm(ref_emb)) 
                               for ref_emb in ref_embeddings]
                prob = max(similarities)  # Use max similarity as score
                print(f"DEBUG: Model {model_data['name']} (Embedding): Score={prob:.4f}")
            else:
                # Standard SVM model
                prob = model.predict_proba([embedding])[0][1]
                print(f"DEBUG: Model {model_data['name']} (SVM): Score={prob:.4f}")
            
            svm_results.append({
                "color_id": color_id,
                "name": model_data['name'],
                "svm_score": prob
            })
        
        # Sort by highest SVM score first
        svm_results = sorted(svm_results, key=lambda x: x['svm_score'], reverse=True)
        print(f"DEBUG: Top 3 SVM results: {[r['name'] + ': ' + str(r['svm_score']) for r in svm_results[:3]]}")
        
        # 3. For the top N SVM matches, run the more expensive LAB comparison
        top_n = min(3, len(svm_results))
        final_results = []
        async with aqlite.connect(DB_FILE) as db:
            for result in svm_results[:top_n]:
                cursor = await db.execute("""
                    SELECT f.lab_features FROM image_features f
                    JOIN training_images i ON f.image_id = i.id
                    WHERE i.color_id = ?
                """, (result['color_id'],))
                
                ref_features = await cursor.fetchall()
                ref_lab_json_list = [row[0] for row in ref_features]

                if not ref_lab_json_list:
                    continue

                # Calculate perceptual color difference score
                lab_score = calculate_lab_similarity(detected_lab_colors, ref_lab_json_list)
                
                # Combine scores (e.g., 70% SVM, 30% LAB). Lower LAB score is better.
                # Normalize LAB score roughly (0-100 scale usually, smaller is better)
                # Let's invert it for combination: 1 / (1 + score)
                combined_score = (result['svm_score'] * 0.7) + ((1 / (1 + lab_score/10.0)) * 0.3)

                result['lab_difference_score'] = lab_score
                result['combined_score'] = combined_score
                final_results.append(result)

        # Sort by the final combined score
        final_results = sorted(final_results, key=lambda x: x['combined_score'], reverse=True)

        # 4. Estimate overall properties
        image_tone, image_level, image_style = estimate_image_properties(detected_lab_colors)

        return {
            "analysis_summary": {
                "estimated_tone": image_tone,
                "estimated_level": image_level,
                "estimated_style": image_style,
            },
            "dominant_hair_colors": display_colors, # Return the display-friendly dicts
            "best_matches": final_results
        }

    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()
        return {"error": "An unexpected server error occurred during analysis."}


async def analyze_image_with_trained_models(images_bytes_list):
    """
    Analyzes one or more images (1-5) individually using all loaded custom models.
    Returns separate analysis result for each image.
    """
    if not loaded_models:
        return {"error": "No trained models are available for analysis."}

    # Handle both single image and list of images
    if not isinstance(images_bytes_list, list):
        images_bytes_list = [images_bytes_list]

    try:
        results = []
        
        # Process each image individually
        for idx, image_bytes in enumerate(images_bytes_list):
            result = await analyze_single_image(image_bytes)
            result['image_index'] = idx + 1  # 1-indexed for user display
            results.append(result)
        
        return {
            "num_images": len(results),
            "results": results
        }

    except Exception as e:
        print(f"An error occurred during multi-image analysis: {e}")
        import traceback
        traceback.print_exc()
        return {"error": "An unexpected server error occurred during analysis."}


