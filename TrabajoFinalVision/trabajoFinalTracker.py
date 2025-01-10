import os
import time
import cv2
import numpy as np
from picamera2 import Picamera2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.7)

def draw_face_box(image, results):
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = image.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0

            # Vamos a calcular el recuadro
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            # Dibujamos el recuadro
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    return image

def main():
    picam = Picamera2()
    picam.preview_configuration.main.size = (1280, 720)
    picam.preview_configuration.main.format = "RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")

    print("Presiona 'e' para salir de la aplicación.")
    picam.start()

    while True:
        frame = picam.capture_array()
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Dibujamos el recuadro alrededor de la cara
        frame = draw_face_box(frame, results)

        cv2.imshow("Vista previa con rastreador facial", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('e'):
            print("Saliendo de la aplicación...")
            break

    picam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
