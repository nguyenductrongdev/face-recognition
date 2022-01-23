import cv2
import pickle

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('model.xml')

faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

face_path = 'C:/Users/Trong Nguyen/Desktop/WIN_20220123_15_14_05_Pro.jpg'

gray = cv2.cvtColor(cv2.imread(face_path), cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(
    100, 100), flags=cv2.CASCADE_SCALE_IMAGE)

le = pickle.load(open('./label_encoder.pkl', 'rb'))

for(x, y, w, h) in faces:
    id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
    print(le.inverse_transform([id]), confidence)

    # print(le.classes_)
