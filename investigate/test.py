import uuid
import cv2
import os
from uuid import uuid4

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def cut_frame(video_path, output_folder_path):
    assert all([video_path, output_folder_path])

    # Opens the Video file
    cap = cv2.VideoCapture(video_path)

    while(cap.isOpened()):
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
        )

        for (x, y, w, h) in faces:
            cv2.imwrite(f"{output_folder_path}/{str(uuid4())}.jpg",
                        frame[y:y + h, x:x + w])

    cap.release()
    cv2.destroyAllWindows()


def main():
    video_paths = [
        os.path.abspath(f'./data/{i}.mp4') for i in range(1, 3+1)
    ]
    outputs = [os.path.abspath('./frame/')]*3

    for path, output in zip(video_paths, outputs):
        cut_frame(path, output)


if __name__ == '__main__':
    main()
