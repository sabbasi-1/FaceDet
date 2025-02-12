# FaceDet
Comparison of faces and detection of multiple faces in a frame

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

  $ Face of John Doe stored successfully.

or if no face is found:

  $ No face found!
