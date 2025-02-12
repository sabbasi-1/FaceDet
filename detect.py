import cv2
from ultralytics import YOLO
import os

model_path = "model.pt"  
model = YOLO(model_path)
image_dir = "input_images"  # Folder containing input images
output_dir = "extracted_faces"  # Folder where cropped faces will be saved
os.makedirs(output_dir, exist_ok=True)
image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    results = model(image)  
    detections = results[0].boxes.xyxy.cpu().numpy() 
    confidences = results[0].boxes.conf.cpu().numpy()  
    class_ids = results[0].boxes.cls.cpu().numpy()  
    face_count = 0
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = map(int, det)  
        confidence = confidences[i]  
        class_id = int(class_ids[i])  
        face = image[y1:y2, x1:x2]
        if face.size > 0:
            face_count += 1
            face_filename = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_face{face_count}.jpg")
            cv2.imwrite(face_filename, face)
            print(f"Saved: {face_filename} (Confidence: {confidence:.2f})")
print("Face extraction complete!")