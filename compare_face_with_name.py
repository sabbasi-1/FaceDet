import cv2
import torch
import pandas as pd
import pickle
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine
from ultralytics import YOLO

CSV_FILE = "face_embeddings.csv"
model = YOLO("model.pt")

def extract_face(image_path):
    image = cv2.imread(image_path)
    results = model(image)
    detections = results[0].boxes.xyxy.cpu().numpy()
    if len(detections) == 0:
        print("No face detected.")
        return None
    x1, y1, x2, y2 = map(int, detections[0])
    face = image[y1:y2, x1:x2]
    if face.size > 0:
        face_resized = cv2.resize(face, (200, 200))
        cv2.imshow("Detected Face", face_resized)
        cv2.waitKey(1)
        return face_resized
    return None

def compare_faces(face_img):
    df = pd.read_csv(CSV_FILE)
    name = input("Enter name to search: ")
    person_row = df[df["name"] == name]
    if person_row.empty:
        print("Name not found in database.")
        return
    hex_str = person_row.iloc[0]["encoding"]
    stored_embedding = pickle.loads(bytes.fromhex(hex_str))
    stored_embedding = np.array(stored_embedding)
    new_embedding = DeepFace.represent(img_path=face_img, model_name="ArcFace", enforce_detection=False)[0]["embedding"]
    new_embedding = np.array(new_embedding)
    cosine_similarity = 1 - cosine(stored_embedding, new_embedding)
    print("Cosine Similarity:", cosine_similarity)
    if cosine_similarity > 0.6:
        print("Face Matched!")
    else:
        print("No Match (Violation)")

face_img = extract_face("2.png")
if face_img is not None:
    compare_faces(face_img)
    cv2.destroyAllWindows()
