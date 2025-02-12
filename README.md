# FaceDet
Comparison of faces and detection of multiple faces in a frame

## Install Dependencies
    pip install -r requirements.txt

## add_face.py
This script extracts face embeddings from an image using DeepFace and saves them along with the person's name in a CSV file (face_embeddings.csv).

🔹 How It Works

1. Extracts face embeddings from the input image using the ArcFace model.

2. Converts the embedding into a hex-encoded string for storage.

3. Checks if face_embeddings.csv exists:

4. If not, it creates one with columns: name and encoding.

5. Appends the new face embedding and name to the CSV file.

6. Prints a success message if the face is stored, or an error if no face is detected.

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

## detect.py

This script detects and extracts faces from images in a specified folder using a trained model and saves them as separate image files.

🔹 How It Works

1. Loads a model (model.pt) trained for face detection.

2. Reads images from input_images/ directory.

3. Runs inference on each image to detect faces.

4. Extracts bounding boxes and crops the detected faces.

5. Saves each detected face in extracted_faces/ directory with a unique filename.

6. Prints the saved face filenames and confidence scores.

🔹 How to Run

1. Prepare the input directory: Place images inside input_images/.

Run the script:

    python extract_faces.py

Extracted faces will be saved in extracted_faces/.

🔹 Expected Output

    Saved: extracted_faces/image1_face1.jpg (Confidence: 0.89)
    Saved: extracted_faces/image1_face2.jpg (Confidence: 0.92)
    Face extraction complete!

If no faces are detected, nothing will be saved.

## add_face_with_name.py

This script detects faces using a model, allows the user to label each face, and then stores the face embeddings in a CSV file using DeepFace.

🔹 How It Works

1. Loads a trained model (model.pt) for face detection.

2. Reads an image (person.jpg) and detects faces.

3. Extracts and resizes each detected face to 200x200 pixels.

4. Displays each detected face to the user.

5. Prompts the user to enter a name for each face.

6. Computes the DeepFace embedding for the face and stores it in face_embeddings.csv.

🔹 How to Run

Run the script:

    python save_face_embeddings.py

The script will:
    
    Detect faces in person.jpg.
    Show each face one by one.
    Ask for a name.
    Save the embedding in face_embeddings.csv.

🔹 Expected Output
    
    [Face is displayed]
    Enter name for this face: John Doe
    Saved: John Doe
    [Face is displayed]
    Enter name for this face: Jane Doe
    Saved: Jane Doe

If no faces are detected, the script will simply exit.

## compare_face_with_name.py

This script detects a face in an image using YOLO, then compares it against stored embeddings in a CSV file using DeepFace and Cosine Similarity.

🔹 How It Works

1. Loads the YOLO model (model.pt) for face detection.

2. Reads an image (2.png) and detects a face.

3. Extracts and resizes the face to 200x200 pixels.

4. Displays the detected face.

5. Prompts the user to enter a name to search for in face_embeddings.csv.

6. Retrieves the stored embedding for the entered name.

7. Extracts embeddings for the detected face using ArcFace.

8. Computes the Cosine Similarity between the stored and new embedding:

9. If similarity > 0.6, the face is considered matched. Otherwise, it's flagged as a violation.

🔹 How to Run

Run the script:
    
    python verify_face.py

The script will:

1. Detect a face in 2.png.

2. Show the face.

3. Ask for a name to search in the database.

4. Compare the detected face with the stored embedding.

🔹 Expected Output

    [Face is displayed]
    Enter name to search: John Doe
    Cosine Similarity: 0.78
    Face Matched!

or if no match is found:

    Enter name to search: John Doe
    Cosine Similarity: 0.45
    No Match (Violation)
    If no face is detected, the script exits.

