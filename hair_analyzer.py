import io
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from ultralytics import YOLO
import cv2

# ----------------------------
# Load the YOLO model
# ----------------------------
model = YOLO('yolov8n-seg.pt')

# ----------------------------
# Predefined Multi-Color Hair Sets
# ----------------------------
HAIR_COLOR_SETS = [
    {
        "name": "Golden Brown Highlights",
        "colors": ["#5A3825", "#B19166", "#E6A575"]
    },
    {
        "name": "Autumn Sunset",
        "colors": ["#6D4C41", "#BF360C", "#FFAB00"]
    },
    {
        "name": "Dark Chocolate and Caramel",
        "colors": ["#3E2723", "#795548", "#FFD54F"]
    },
    {
        "name": "Beachy Blonde",
        "colors": ["#F0EAD6", "#D2B48C", "#FFF8E1"]
    },
    {
        "name": "Silver Fox",
        "colors": ["#4A4A4A", "#A9A9A9", "#E0E0E0"]
    },
    {
        "name": "Burgundy Bliss",
        "colors": ["#5D1A2A", "#8C2B3F", "#C04D5F"]
    },
    {
        "name": "Ashy Brown",
        "colors": ["#594D44", "#8C7B6E", "#BFAE9F"]
    },
    {
        "name": "Strawberry Blonde",
        "colors": ["#A56B46", "#D49A7A", "#F8D6B2"]
    },
    {
        "name": "Jet Black",
        "colors": ["#0A0A0A", "#1C1C1C", "#2E2E2E"]
    }
]

# ----------------------------
# Helper function: Hex to RGB
# ----------------------------
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# ----------------------------
# Helper function: Color difference in LAB space
# ----------------------------
def color_difference_lab(rgb1, rgb2):
    # Convert RGB to LAB
    lab1 = cv2.cvtColor(np.uint8([[rgb1]]), cv2.COLOR_RGB2LAB)[0][0]
    lab2 = cv2.cvtColor(np.uint8([[rgb2]]), cv2.COLOR_RGB2LAB)[0][0]
    # Calculate Euclidean distance
    return np.linalg.norm(lab1.astype(float) - lab2.astype(float))

# ----------------------------
# Hair Segmentation
# ----------------------------
def segment_hair(image: Image.Image) -> np.ndarray:
    results = model(image)
    
    if results[0].masks is None or len(results[0].masks.data) == 0:
        return None

    masks = results[0].masks.data.cpu().numpy()
    # Assuming the largest mask is most relevant
    hair_mask = np.sum(masks, axis=0)
    hair_mask = np.clip(hair_mask, 0, 1)
    
    hair_mask = cv2.resize(hair_mask, (image.width, image.height))
    
    return hair_mask

# ----------------------------
# Get Dominant Colors from Hair
# ----------------------------
def get_dominant_colors(image: Image.Image, hair_mask: np.ndarray, n_colors=5) -> list:
    np_image = np.array(image)
    
    hair_pixels = np_image[hair_mask > 0.5] # Use a threshold
    
    if len(hair_pixels) < n_colors:
        return []

    kmeans = KMeans(n_clusters=n_colors, n_init='auto', random_state=42)
    kmeans.fit(hair_pixels)

    dominant_colors = kmeans.cluster_centers_.astype(int)
    
    counts = np.bincount(kmeans.labels_)
    sorted_indices = np.argsort(counts)[::-1]
    
    dominant_colors = dominant_colors[sorted_indices]
    
    hex_colors = [f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}" for rgb in dominant_colors]
    
    return hex_colors

# ----------------------------
# Find Closest Hair Color Sets
# ----------------------------
def find_closest_hair_color_sets(detected_colors: list, color_sets: list, top_n=3) -> list:
    set_distances = []

    detected_colors_rgb = [hex_to_rgb(c) for c in detected_colors]

    for color_set in color_sets:
        set_colors_rgb = [hex_to_rgb(c) for c in color_set["colors"]]
        
        total_min_distance = 0
        for detected_color in detected_colors_rgb:
            min_dist = float('inf')
            for set_color in set_colors_rgb:
                dist = color_difference_lab(detected_color, set_color)
                if dist < min_dist:
                    min_dist = dist
            total_min_distance += min_dist
        
        avg_distance = total_min_distance / len(detected_colors_rgb)
        set_distances.append({"set": color_set, "distance": avg_distance})

    # Sort sets by distance
    sorted_sets = sorted(set_distances, key=lambda x: x['distance'])

    # Prepare the top N results
    top_matches = []
    # Max possible distance in LAB space is ~sqrt(100^2 + 2*128^2) but we can normalize differently
    # A simpler normalization: use the max distance found as the upper bound
    max_found_distance = sorted_sets[-1]['distance'] if sorted_sets else 1

    for match in sorted_sets[:top_n]:
        # Normalize similarity based on the range of distances found
        similarity = 100 * (1 - match['distance'] / (max_found_distance * 1.5)) # Add buffer
        similarity = max(0, min(similarity, 100)) # Clamp between 0 and 100

        top_matches.append({
            "set_name": match["set"]["name"],
            "similarity_percentage": f"{similarity:.2f}%",
            "matched_colors": match["set"]["colors"]
        })
        
    return top_matches

# ----------------------------
# Main analysis function
# ----------------------------
def analyze_image_color(image_bytes: bytes) -> dict:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        hair_mask = segment_hair(image)
        if hair_mask is None:
            return {"error": "Could not segment hair from the image."}

        dominant_colors = get_dominant_colors(image, hair_mask, n_colors=5)
        if not dominant_colors:
            return {"error": "Could not extract dominant colors from the hair region."}

        closest_sets = find_closest_hair_color_sets(dominant_colors, HAIR_COLOR_SETS, top_n=3)

        return {
            "dominant_hair_colors": dominant_colors,
            "closest_hair_color_sets": closest_sets
        }
    except Exception as e:
        return {"error": f"Failed to analyze image: {str(e)}"}
