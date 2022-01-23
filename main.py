import cv2
import pickle

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

le = pickle.load(open('./label_encoder.pkl', 'rb'))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('model.xml')

while True:
    ret, im = cam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(
        100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x, y, w, h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if(confidence < 100):
            id, *_ = le.inverse_transform([id])

            confidence = "  {0}%".format(round(100 - confidence))
            # print("Khải Tk")
        else:
            id = "Unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(im, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(im, str(confidence), (x + 5, y + h - 5),
                    font, 1, (255, 255, 0), 1)

    out.write(im)
    cv2.imshow('im', im)
    cv2.waitKey(10)
cap.release()
out.release()
cv2.destroyAllWindows()
