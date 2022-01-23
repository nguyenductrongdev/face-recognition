from PIL import Image
from numpy import asarray, dtype
import numpy as np
import os
import json
import cv2
from sklearn import preprocessing
import pickle

FACE_FOLDERS_PATH = os.path.abspath('./data/face/')

list_subfolders_with_paths = [
    f.path
    for f in os.scandir(FACE_FOLDERS_PATH)
    if f.is_dir()
]


X = []
y = []

for face_folder in list_subfolders_with_paths:
    face_images = [f.path for f in os.scandir(face_folder) if f.is_file()]
    for image_path in face_images:
        # load the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # convert image to numpy array
        data = asarray(image, dtype=np.uint8)
        label = face_folder.split('\\')[-1]

        X.append(data)
        y.append(label)

le = preprocessing.LabelEncoder()
le.fit(y)

model = cv2.face.LBPHFaceRecognizer_create()

model.train(X, le.transform(y))
model.save('model.xml')

pickle.dump(le, open("label_encoder.pkl", "wb"))

print("Model Training Complete!!!!!")
