from flask import Flask, request, render_template, send_from_directory
import face_recognition
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load embeddings từ file face_vectors.pkl
with open("face_vectors.pkl", "rb") as f:
    face_vectors = pickle.load(f)

# Hàm tìm người giống nhất
def find_best_match(img_path, threshold=0.95):
    img = face_recognition.load_image_file(img_path)
    encodings = face_recognition.face_encodings(img)

    if not encodings:
        return "No face detected", 0.0

    query_vec = encodings[0]
    best_score = -1
    best_name = "Unknown"

    for name, vec in face_vectors.items():
        score = cosine_similarity([query_vec], [vec])[0][0]
        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= threshold:
        return best_name, best_score
    else:
        return "Unknown", best_score

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('file')  # Get list of files
        if not files:
            return render_template('index.html', error='No files selected')

        results = []
        for file in files:  # Process up to 10 files
            if file.filename == '':  # Ignore empty filenames
                continue

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Get result for each image
            result, score = find_best_match(filepath)
            results.append({
                'filename': file.filename,
                'result': result,
                'score': score
            })

        return render_template('index.html', results=results)

    return render_template('index.html', results=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, use_reloader=False)
