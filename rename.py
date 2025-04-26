import os

def rename_images_in_all_subfolders(root_folder):
    # Lặp qua tất cả các thư mục con
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)

        # Kiểm tra nếu là thư mục
        if os.path.isdir(subfolder_path):
            files = sorted(os.listdir(subfolder_path))
            count = 1

            for file_name in files:
                old_path = os.path.join(subfolder_path, file_name)

                # Chỉ rename nếu là file ảnh
                if os.path.isfile(old_path) and file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    ext = os.path.splitext(file_name)[1]  # Lấy đuôi file .jpg .png...
                    new_name = f"img{count}{ext}"
                    new_path = os.path.join(subfolder_path, new_name)

                    os.rename(old_path, new_path)
                    print(f"Renamed {old_path} -> {new_path}")

                    count += 1

# Example usage
rename_images_in_all_subfolders('database_faces')  # <-- Thay bằng đường dẫn folder gốc của bạn
