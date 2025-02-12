from deepface import DeepFace
import numpy as np
import pandas as pd
import pickle
from scipy.spatial.distance import cosine

CSV_FILE = "face_embeddings.csv"
df = pd.read_csv(CSV_FILE)
hex_str = df.loc[0, "encoding"]  
embedding_array = pickle.loads(bytes.fromhex(hex_str))  
print("Recovered Face Embedding:", embedding_array)
print("Shape of embedding:", np.array(embedding_array).shape)   
stored_embedding = np.array(embedding_array)
new_embedding = DeepFace.represent(img_path="2.png", model_name="ArcFace")[0]["embedding"]
new_embedding = np.array(new_embedding)
cosine_similarity = 1 - cosine(stored_embedding, new_embedding)
print("Cosine Similarity:", cosine_similarity)
if cosine_similarity > 0.6:
    print("Face Matched!")
else:
    print("No Match (Violation)")
