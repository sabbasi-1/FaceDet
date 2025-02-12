from deepface import DeepFace
import pandas as pd
import pickle
import numpy as np
import os
CSV_FILE = "face_embeddings.csv"

def add_face(image_path, name):
    embedding = DeepFace.represent(img_path=image_path, model_name="ArcFace", enforce_detection=True)
    
    if embedding:
        encoding_array = embedding[0]['embedding']  
        encoding_str = pickle.dumps(encoding_array).hex()  
        if not os.path.exists(CSV_FILE):
            df = pd.DataFrame(columns=["name", "encoding"])
            df.to_csv(CSV_FILE, index=False)

        df = pd.read_csv(CSV_FILE)
        df = pd.concat([df, pd.DataFrame([{"name": name, "encoding": encoding_str}])], ignore_index=True)
        df.to_csv(CSV_FILE, index=False)

        print(f"Face of {name} stored successfully.")
    else:
        print("No face found!")
add_face("person.jpg", "John Doe")
