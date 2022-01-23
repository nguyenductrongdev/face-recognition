import cv2
import os
from uuid import uuid4

CLASS_FOLDERS = os.path.abspath('./data/raw/')

list_subfolders_with_paths = [
    f.path
    for f in os.scandir(CLASS_FOLDERS)
    if f.is_dir()
]

for class_folder in list_subfolders_with_paths:
    images = [f.path for f in os.scandir(class_folder) if f.is_file()]
    for image_path in images:
        # Bước 1: Tấm ảnh và tệp tin xml
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        image = cv2.imread(image_path)

        # Bước 2: Tạo một bức ảnh xám
        grayImage = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)

        # Bước 3: Tìm khuôn mặt
        faces = face_cascade.detectMultiScale(
            grayImage,
            scaleFactor=1.1,
            minNeighbors=5,
        )

        current_class = class_folder.split("\\")[-1]
        face_folder = f'./data/face/{current_class}'
        face_path = os.path.abspath(
            f'{face_folder}/{str(uuid4())}.jpg')

        if not os.path.exists(face_folder):
            os.mkdir(face_folder)
        # Bước 4: Vẽ các khuôn mặt đã nhận diện được lên tấm ảnh gốc
        for (x, y, w, h) in faces:
            cv2.imwrite(face_path, image[y:y + h, x:x + w])
