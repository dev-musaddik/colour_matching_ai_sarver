import io
import cv2
import torch
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# ----------------------------
# Load YOLOv8 TorchScript model
# ----------------------------
# Make sure you export your YOLOv8 segmentation model first:
# yolo export model=yolov8n-seg.pt format=torchscript
MODEL_PATH = "yolov8n-seg.ts"
model = torch.jit.load(MODEL_PATH, map_location="cpu")
model.eval()  # set to evaluation mode

# ----------------------------
# Helper function: Dominant color
# ----------------------------
def get_dominant_color(image: Image.Image) -> tuple:
    image = image.resize((100, 100))
    np_image = np.array(image)
    np_image = np_image.reshape((-1, 3))

    # Remove black pixels
    non_black_pixels = np_image[np.any(np_image != [0, 0, 0], axis=1)]
    if len(non_black_pixels) == 0:
        return (0, 0, 0), "#000000"

    n_clusters = min(3, len(non_black_pixels))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(non_black_pixels)

    counts = np.bincount(kmeans.labels_)
    dominant_cluster = np.argmax(counts)
    dominant_color = kmeans.cluster_centers_[dominant_cluster].astype(int)

    rgb_color = tuple(int(c) for c in dominant_color)
    hex_color = "#%02x%02x%02x" % rgb_color
    return rgb_color, hex_color

# ----------------------------
# Main function: Analyze hair color
# ----------------------------
def analyze_hair_color(image_bytes: bytes) -> dict:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_image = np.array(image)

    # YOLOv8 TorchScript expects torch.Tensor
    input_tensor = torch.from_numpy(np_image).permute(2, 0, 1).unsqueeze(0).float()
    input_tensor /= 255.0  # normalize

    with torch.no_grad():
        outputs = model(input_tensor)  # model returns a dict with masks/boxes

    # Extract masks and boxes
    masks = outputs[0]["masks"].numpy() if "masks" in outputs[0] else None
    boxes = outputs[0]["boxes"].numpy() if "boxes" in outputs[0] else None
    classes = outputs[0]["classes"].numpy() if "classes" in outputs[0] else None

    if masks is None or boxes is None or classes is None:
        return {"error": "No person detected."}

    # Filter person masks
    person_masks = []
    person_boxes = []
    for i, cls in enumerate(classes):
        if int(cls) == 0:  # 'person' class in COCO is 0
            person_masks.append(masks[i])
            person_boxes.append(boxes[i])

    if not person_masks:
        return {"error": "No person detected."}

    # Combine masks
    combined_mask = np.sum(person_masks, axis=0)
    combined_mask = (combined_mask > 0).astype(np.uint8)
    combined_mask = cv2.resize(combined_mask, (np_image.shape[1], np_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Take first person box for hair region
    box = person_boxes[0]
    x1, y1, x2, y2 = map(int, box)
    hair_region_y_end = y1 + int((y2 - y1) * 0.25)

    hair_mask = np.zeros_like(combined_mask)
    hair_mask[y1:hair_region_y_end, x1:x2] = 1
    final_hair_mask = combined_mask * hair_mask

    if np.sum(final_hair_mask) == 0:
        return {"error": "Could not isolate hair region."}

    hair_only_image = cv2.bitwise_and(np_image, np_image, mask=final_hair_mask)
    hair_pil_image = Image.fromarray(hair_only_image)

    rgb_color, hex_color = get_dominant_color(hair_pil_image)

    return {
        "dominant_color_rgb": rgb_color,
        "dominant_color_hex": hex_color,
    }

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    with open("test_image.jpg", "rb") as f:
        image_bytes = f.read()

    result = analyze_hair_color(image_bytes)
    print(result)
