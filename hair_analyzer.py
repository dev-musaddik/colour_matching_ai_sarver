import io
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from ultralytics import YOLO
import cv2
from skimage.color import rgb2lab, deltaE_ciede2000
from typing import List, Dict, Any, Tuple

# -----------------------------
# Universal JSON-safe converter
# -----------------------------
def to_serializable(obj):
    """Converts numpy types into JSON-serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(to_serializable(v) for v in obj)
    return obj

# Load the YOLO model for hair segmentation
model = YOLO('yolov8n-seg.pt')


def segment_hair(image: Image.Image) -> np.ndarray:
    """Segments hair from an image using YOLOv8 and returns a binary mask."""
    results = model(image, verbose=False)

    if not results or results[0].masks is None or len(results[0].masks.data) == 0:
        return None

    masks = results[0].masks.data.cpu().numpy()
    hair_mask = np.sum(masks, axis=0)
    hair_mask = np.clip(hair_mask, 0, 1)

    return cv2.resize(hair_mask, (image.width, image.height), interpolation=cv2.INTER_NEAREST)


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

    # Make response JSON-safe
    response = {
        "lab_signature": lab_signature,
        "display_colors": display_colors
    }

    return to_serializable(response)
