import os
import face_recognition
import pickle

def create_embeddings(dataset_path='dataset', output_path='face_vectors.pkl'):
    face_vectors = {}

    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue

        person_encodings = []
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            img = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(img)

            if encodings:
                person_encodings.append(encodings[0])  # Lấy encoding đầu tiên
            else:
                print(f"Warning: No face found in {img_path}")
        if person_encodings:
            face_vectors[person_name] = person_encodings

    # Lưu lại tất cả vectors
    with open(output_path, 'wb') as f:
        pickle.dump(face_vectors, f)

    print(f"Saved embeddings for {len(face_vectors)} people to {output_path}")

# --- GỌI ---
create_embeddings("database_faces")
