# FaceDet
Comparison of faces and detection of multiple faces in a frame

## Install Dependencies
    pip install -r requirements.txt

## add_face.py
This script extracts face embeddings from an image using DeepFace and saves them along with the person's name in a CSV file (face_embeddings.csv).

🔹 How It Works

Extracts face embeddings from the input image using the ArcFace model.
Converts the embedding into a hex-encoded string for storage.
Checks if face_embeddings.csv exists:
If not, it creates one with columns: name and encoding.
Appends the new face embedding and name to the CSV file.
Prints a success message if the face is stored, or an error if no face is detected.

🔹 How to Run

Ensure you have installed dependencies:

    pip install deepface pandas numpy pickle-mixin

Run the script with an image path and a name:

    python add_face.py

The script will process person.jpg and store the embedding under the name "John Doe".

🔹 Expected Output
    
    Face of John Doe stored successfully.

or if no face is found:
    
    No face found!

## compare.py

This script compares a detected face with stored embeddings in face_embeddings.csv using DeepFace and Cosine Similarity.

🔹 How It Works

1. Loads stored face embeddings from face_embeddings.csv.

2. Retrieves the first stored face embedding and converts it from a hex string back to a NumPy array.

3. Extracts embeddings from a new image (2.png) using DeepFace's ArcFace model.

4. Computes the Cosine Similarity between the stored embedding and the new image.

5. Prints similarity score and determines a match:

6. If similarity > 0.6, the face is considered matched. Otherwise, it is flagged as a violation.
   
🔹 How to Run

Run the script:

    python compare_face.py

The script will compare the detected face in 2.png with the first stored embedding.

🔹 Expected Output

    Recovered Face Embedding: [array of numbers]
    Shape of embedding: (128,)
    Cosine Similarity: 0.72
    Face Matched!

or if no match is found:

    Cosine Similarity: 0.45
    No Match (Violation)
