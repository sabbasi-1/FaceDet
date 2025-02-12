import cv2
import torch
import pandas as pd
import pickle
import numpy as np
import os
from deepface import DeepFace
from ultralytics import YOLO

CSV_FILE = "face_embeddings.csv"
model = YOLO("model.pt")

def save_embedding(face_img, name):
    embedding = DeepFace.represent(img_path=face_img, model_name="ArcFace", enforce_detection=True)
    if embedding:
        encoding_str = pickle.dumps(embedding[0]['embedding']).hex()
        if not os.path.exists(CSV_FILE):
            df = pd.DataFrame(columns=["name", "encoding"])
            df.to_csv(CSV_FILE, index=False)
        df = pd.read_csv(CSV_FILE)
        df = pd.concat([df, pd.DataFrame([{"name": name, "encoding": encoding_str}])], ignore_index=True)
        df.to_csv(CSV_FILE, index=False)
        print(f"Saved: {name}")

def extract_faces(image_path):
    image = cv2.imread(image_path)
    results = model(image)
    detections = results[0].boxes.xyxy.cpu().numpy()
    faces = []
    for det in detections:
        x1, y1, x2, y2 = map(int, det)
        face = image[y1:y2, x1:x2]
        if face.size > 0:
            face_resized = cv2.resize(face, (200, 200))
            faces.append(face_resized)
    return faces

def main(image_path):
    faces = extract_faces(image_path)
    for face in faces:
        cv2.imshow("Detected Face", face)
        cv2.waitKey(1)
        name = input("Enter name for this face: ")
        save_embedding(face, name)
    cv2.destroyAllWindows()

main("person.jpg")
