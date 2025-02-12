import cv2
from ultralytics import YOLO
import os

# Load the trained YOLO model
model_path = "model.pt"  # Path to your trained YOLO model
model = YOLO(model_path)

# Define directories
image_dir = "input_images"  # Folder containing input images
output_dir = "extracted_faces"  # Folder where cropped faces will be saved

# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)

# Get list of all images
image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))]

# Process each image
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)

    # Load image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Run YOLO face detection
    results = model(image)  # YOLOv8 inference

    # Extract bounding boxes from YOLOv8 results
    detections = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = results[0].boxes.cls.cpu().numpy()  # Class labels

    face_count = 0
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = map(int, det)  # Extract bbox coordinates
        confidence = confidences[i]  # Confidence score
        class_id = int(class_ids[i])  # Class label (if multi-class)

        # Crop the face
        face = image[y1:y2, x1:x2]

        # Save the face if it's not an empty crop
        if face.size > 0:
            face_count += 1
            face_filename = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_face{face_count}.jpg")
            cv2.imwrite(face_filename, face)
            print(f"Saved: {face_filename} (Confidence: {confidence:.2f})")

print("✅ Face extraction complete!")