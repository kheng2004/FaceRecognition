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

from sklearn.metrics.pairwise import cosine_similarity

# Hàm tìm người giống nhất
def find_best_match(img_path, threshold=0.94, margin=0.02):
    img = face_recognition.load_image_file(img_path)
    encodings = face_recognition.face_encodings(img)

    if not encodings:
        return "No face detected", 0.0

    query_vec = encodings[0]
    similarities = []

    for name, vec_list in face_vectors.items():
        # vec_list là danh sách vectors (1 người nhiều vector)
        for vec in vec_list:
            score = cosine_similarity([query_vec], [vec])[0][0]
            similarities.append((name, score))

    # Sắp xếp theo điểm cosine giảm dần
    similarities.sort(key=lambda x: x[1], reverse=True)

    top1_name, top1_score = similarities[0]
    # Tìm top2: khác class với top1 (nếu có)
    top2_score = 0.0
    for name, score in similarities[1:]:
        if name != top1_name:
            top2_score = score
            break

    # Kiểm tra điều kiện nhận dạng
    if (top1_score >= threshold) and ((top1_score - top2_score) >= margin):
        return top1_name, top1_score
    else:
        return "Unknown", top1_score



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
