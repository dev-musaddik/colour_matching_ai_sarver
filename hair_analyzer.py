import io
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import List, Dict, Any, Tuple
from skimage.color import rgb2lab, deltaE_ciede2000
import cv2

# -----------------------------
# MediaPipe Setup
# -----------------------------
mp_selfie_segmentation = mp.solutions.selfie_segmentation

def segment_hair(image: Image.Image) -> np.ndarray:
    """
    Segments hair/person from an image using MediaPipe Selfie Segmentation.
    Returns a binary mask (1 for hair/person, 0 for background).
    
    Note: MediaPipe Selfie Segmentation segments the *person*, not just hair.
    However, for hair color analysis, the dominant colors in the person mask 
    (excluding skin tones via color logic if needed, but usually hair is dominant enough)
    works well as a lightweight proxy.
    """
    # Convert PIL Image to NumPy array (RGB)
    image_np = np.array(image)

    # Initialize MediaPipe Selfie Segmentation
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        # Process the image
        results = selfie_segmentation.process(image_np)
        
        if results.segmentation_mask is None:
            return None

        # The mask is a float array [0, 1]. We threshold it.
        # > 0.5 is considered person/hair
        mask = results.segmentation_mask
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        return binary_mask


def extract_color_signature(image: Image.Image, hair_mask: np.ndarray, n_colors=3) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """
    Extracts the top N dominant colors from the hair region using k-means.
    Returns both LAB vectors for matching and RGB/hex info for display.
    """
    np_image = np.array(image)
    mask_bool = hair_mask > 0.5
    hair_pixels = np_image[mask_bool]

    if len(hair_pixels) < n_colors * 10:
        return None, None

    sample_pixels = hair_pixels[np.random.choice(len(hair_pixels), size=min(10000, len(hair_pixels)), replace=False)]

    kmeans = KMeans(n_clusters=n_colors, n_init='auto', random_state=42)
    kmeans.fit(sample_pixels)

    counts = np.bincount(kmeans.predict(hair_pixels))
    total_pixels = len(hair_pixels)

    sorted_indices = np.argsort(counts)[::-1]
    dominant_colors_rgb = kmeans.cluster_centers_.astype(int)[sorted_indices]
    percentages = (counts[sorted_indices] / total_pixels) * 100

    dominant_colors_lab = [
        rgb2lab(c.reshape(1, 1, 3) / 255.0).reshape(3).astype(float)
        for c in dominant_colors_rgb
    ]

    display_colors = [
        {
            "hex": f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}",
            "rgb": (int(rgb[0]), int(rgb[1]), int(rgb[2])),
            "percentage": float(round(p, 2))
        }
        for rgb, p in zip(dominant_colors_rgb, percentages)
    ]

    return dominant_colors_lab, display_colors


def find_closest_match(user_lab_signature: List[np.ndarray], trained_colors: List[Dict]) -> Dict:
    """
    Finds the best match for a user's hair color signature from the trained dataset.
    Uses a weighted CIEDE2000 distance calculation.
    """
    if not trained_colors:
        return None

    best_match = None
    lowest_distance = float('inf')

    for trained_color in trained_colors:
        trained_lab_signature = [np.array(c) for c in trained_color["lab_colors"]]

        distance = 0
        num_colors = min(len(user_lab_signature), len(trained_lab_signature))
        for i in range(num_colors):
            distance += deltaE_ciede2000(user_lab_signature[i], trained_lab_signature[i])

        avg_distance = distance / num_colors if num_colors > 0 else float('inf')

        if avg_distance < lowest_distance:
            lowest_distance = avg_distance
            best_match = trained_color

    if best_match:
        similarity = max(0, 100 - lowest_distance * 2.5)
        return {
            "name": best_match["name"],
            "distance": float(round(lowest_distance, 2)),
            "similarity": float(round(similarity, 2)),
            "source_image": best_match.get("source_image")
        }

    return None


def analyze_hair_color_from_image(image_bytes: bytes) -> Dict:
    """
    Full pipeline: segment hair, extract color signature, and return results.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return {"error": "Invalid image file."}

    hair_mask = segment_hair(image)
    if hair_mask is None or np.sum(hair_mask) < 500:
        return {"error": "Could not detect a significant hair region. Please use a clearer photo."}

    lab_signature, display_colors = extract_color_signature(image, hair_mask)
    if lab_signature is None:
        return {"error": "Could not determine dominant colors from the detected hair."}

    response = {
        "lab_signature": lab_signature,
        "display_colors": display_colors
    }

    return to_serializable(response)


# Backward compatibility aliases
def analyze_image_color(image_bytes: bytes) -> Dict:
    """Legacy function - calls analyze_hair_color_from_image."""
    result = analyze_hair_color_from_image(image_bytes)
    if "error" not in result:
        return {
            "user_hair_colors": result.get("display_colors", []),
            "lab_signature": result.get("lab_signature", [])
        }
    return result


def get_dominant_colors(image: Image.Image, hair_mask: np.ndarray, n_colors: int = 3):
    """Alias for extract_color_signature - used by training_service."""
    lab_colors, _ = extract_color_signature(image, hair_mask, n_colors)
    return lab_colors


def color_difference_lab(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """Calculate color difference using CIEDE2000."""
    return float(deltaE_ciede2000(lab1, lab2))


def estimate_image_properties(lab_colors: List[np.ndarray]) -> Dict[str, Any]:
    """Estimate hair properties based on LAB color values."""
    if not lab_colors or len(lab_colors) == 0:
        return {
            "estimated_tone": "Unknown",
            "estimated_level": "Unknown",
            "estimated_style": "Unknown"
        }
    
    dominant_lab = lab_colors[0]
    L, a, b = dominant_lab[0], dominant_lab[1], dominant_lab[2]
    
    # Estimate level
    if L > 70:
        level = "Very Light"
    elif L > 50:
        level = "Light"
    elif L > 30:
        level = "Medium"
    else:
        level = "Dark"
    
    # Estimate tone
    if abs(a) < 5 and abs(b) < 5:
        tone = "Neutral"
    elif a > 10:
        tone = "Warm/Red"
    elif a < -5:
        tone = "Cool/Green"
    elif b > 10:
        tone = "Golden/Yellow"
    elif b < -5:
        tone = "Ash/Blue"
    else:
        tone = "Neutral"
    
    # Estimate style
    if L > 60:
        style = "Blonde"
    elif L > 40:
        style = "Brown"
    else:
        style = "Black"
    
    return {
        "estimated_tone": tone,
        "estimated_level": level,
        "estimated_style": style
    }
