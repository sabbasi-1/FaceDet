import pandas as pd
import pickle
import numpy as np

# Load the CSV file
CSV_FILE = "face_embeddings.csv"
df = pd.read_csv(CSV_FILE)

# Convert first encoding back to a NumPy array
hex_str = df.loc[0, "encoding"]  # Get the first stored embedding
embedding_array = pickle.loads(bytes.fromhex(hex_str))  # Convert hex to bytes and deserialize

print("Recovered Face Embedding:", embedding_array)
print("Shape of embedding:", np.array(embedding_array).shape)  #