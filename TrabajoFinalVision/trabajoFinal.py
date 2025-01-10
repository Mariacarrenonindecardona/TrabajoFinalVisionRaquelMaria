import os
import time
import cv2
import numpy as np
from picamera2 import Picamera2
import mediapipe as mp

# Comenzamos utilizando la librería Mediapipe para la detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

# Utilizamos la librería Mediapipe para la detección facial
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.7)

def is_finger_extended(tip, dip, pip):
    return tip.y < dip.y < pip.y  # El dedo está extendido si sigue esta relación

def detect_gesture(image):
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
        return "Piedra"  # Si no se detecta mano, asumimos "Piedra"

    landmarks = results.multi_hand_landmarks[0].landmark

    # Vamos a tomar las coordenadas de las articulaciones de los dedos
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]

    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_dip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]

    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_dip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    middle_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]

    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_dip = landmarks[mp_hands.HandLandmark.RING_FINGER_DIP]
    ring_pip = landmarks[mp_hands.HandLandmark.RING_FINGER_PIP]

    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    pinky_dip = landmarks[mp_hands.HandLandmark.PINKY_DIP]
    pinky_pip = landmarks[mp_hands.HandLandmark.PINKY_PIP]

    thumb_extended = is_finger_extended(thumb_tip, thumb_ip, thumb_mcp)
    index_extended = is_finger_extended(index_tip, index_dip, index_pip)
    middle_extended = is_finger_extended(middle_tip, middle_dip, middle_pip)
    ring_extended = is_finger_extended(ring_tip, ring_dip, ring_pip)
    pinky_extended = is_finger_extended(pinky_tip, pinky_dip, pinky_pip)

    if thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
        return "Piedra" #Si los dedos que se ven no están extendidos -> piedra
    elif index_extended and middle_extended and not ring_extended and not pinky_extended:
        return "Tijeras" # Dedos índice y medio extendidos -> tijeras
    elif all([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]):
        return "Papel" # Todos los dedos están extendidos -> papel
    else:
        return None  # Gesto no identificado

# Función para superponer filtros en la imagen
def overlay_image(background, overlay, position, size=None):
    x, y = position
    if size:
        overlay = cv2.resize(overlay, size, interpolation=cv2.INTER_AREA)

    h, w, _ = overlay.shape
    bg_h, bg_w, _ = background.shape

    # Validación para asegurar que la posición y el tamaño sean válidos
    if y < 0 or x < 0 or y >= bg_h or x >= bg_w:
        print("El filtro está fuera de los límites de la imagen. No se puede aplicar.")
        return background  # No se puede superponer fuera de los límites, así evitamos que de error.

    h = min(h, bg_h - max(0, y))
    w = min(w, bg_w - max(0, x))
    overlay = overlay[:h, :w]

    if overlay.shape[2] < 4:  # Si la imagen no tiene canal alfa, no se puede usar
        print("La imagen del filtro no tiene canal alfa.")
        return background

    # Nos aseguramos de no usar índices negativos
    y = max(0, y)
    x = max(0, x)

    alpha_overlay = overlay[:, :, 3] / 255.0
    alpha_background = 1.0 - alpha_overlay

    for c in range(3):
        background[y:y+h, x:x+w, c] = (
            alpha_overlay * overlay[:, :, c] + alpha_background * background[y:y+h, x:x+w, c]
        )
    return background

def apply_filters(photo, filter_key, filter_img):
    results = face_mesh.process(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        print("No se detectó ninguna cara. Eliminando filtro.")
        return photo  # Si no se detectan caras, devuelve la foto original

    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = photo.shape

    #Vamos a calcular los puntos de referencia
    try:
        left_eye = (int(landmarks[33].x * w), int(landmarks[33].y * h))
        right_eye = (int(landmarks[263].x * w), int(landmarks[263].y * h))
        nose = (int(landmarks[1].x * w), int(landmarks[1].y * h))
        forehead = (int(landmarks[10].x * w), int(landmarks[10].y * h))
    except IndexError:
        print("Error al calcular los puntos de referencia. Eliminando filtro.")
        return photo

    if filter_key == 'm':  # Bigote
        position = (nose[0] - w // 8, nose[1] - h // 20)  # Ajustamos la posición para que esté más arriba
        size = (w // 4, h // 8)
    elif filter_key == 'g':  # Gafas
        mid_eye_x = (left_eye[0] + right_eye[0]) // 2
        mid_eye_y = (left_eye[1] + right_eye[1]) // 2
        position = (mid_eye_x - w // 6, mid_eye_y - h // 20)
        size = (w // 3, h // 8)
    elif filter_key == 'h':  # Sombrero
        position = (forehead[0] - w // 4, forehead[1] - h // 6)
        size = (w // 2, h // 4)
    else:
        print("Filtro no válido.")
        return photo

    # Verificar si las coordenadas calculadas están dentro de los límites de la imagen
    if position[0] < 0 or position[1] < 0 or position[0] >= w or position[1] >= h:
        #print("El filtro está fuera de los límites de la imagen. No se puede aplicar.")
        return photo

    return overlay_image(photo, filter_img, position, size)

def main():
    picam = Picamera2()
    picam.preview_configuration.main.size = (1280, 720)
    picam.preview_configuration.main.format = "RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")

    order = ["Piedra", "Tijeras", "Papel"]
    detected_sequence = []

    print("Realiza la secuencia: Piedra, Tijeras, Papel. Pulsa la barra espaciadora para registrar cada gesto.")
    picam.start()

    while len(detected_sequence) < 3:
        frame = picam.capture_array()
        cv2.imshow("Vista previa", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # Barra espaciadora
            gesture = detect_gesture(frame)
            expected = order[len(detected_sequence)]
            if gesture == expected:
                detected_sequence.append(gesture)
                print(f"{gesture} detectado correctamente.")
            else:
                print(f"Secuencia incorrecta. Se esperaba {expected}, pero se detectó {gesture}.")
                detected_sequence_str = " -> ".join(detected_sequence + [gesture])
                print(f"Secuencia obtenida: {detected_sequence_str}")
                picam.stop()
                cv2.destroyAllWindows()
                print("Reiniciando en 4 segundos...")
                time.sleep(4)
                detected_sequence = []
                picam.start()

    picam.stop()
    cv2.destroyAllWindows()
    print("¡Enhorabuena! Has accedido correctamente.")
    print("Ahora vamos a pasar a los filtros. Puedes probar gafas (tecla 'g'), bigote (tecla 'm') o sombrero (tecla 'h'). Además puedes utilizar la tecla 'w' para verlo en blanco y negro, y la tecla 'b' para verlo en beige. Además puedes presionar la barra espaciadora para tomar una foto que luego se guardará en la carpeta 'fotos_tomadas'. ")
    input("Presiona Enter para continuar.")

    # Segunda fase: Filtros
    picam.start()

    filter_folder = "./filters"
    filter_files = {'h': "sombrero.png", 'g': "gafas.png", 'm': "bigote.png"}
    current_filter = None
    filter_img = None
    current_mode = 'n'  # Modo actual de visualización: 'n' normal, 'w' blanco y negro, 'b' beige

    output_folder = "fotos_tomadas"
    os.makedirs(output_folder, exist_ok=True)

    while True:
        frame = picam.capture_array()

        if current_mode == 'w':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif current_mode == 'b':
            beige_filter = np.full(frame.shape, (200, 180, 150), dtype=np.uint8)
            frame = cv2.addWeighted(frame, 0.5, beige_filter, 0.5, 0)

        if filter_img is not None:
            frame = apply_filters(frame.copy(), current_filter, filter_img)

        cv2.imshow("Vista previa con filtro", frame)

        key = cv2.waitKey(1) & 0xFF

        if key in [ord('h'), ord('g'), ord('m')]:
            current_filter = chr(key)
            filter_img = cv2.imread(os.path.join(filter_folder, filter_files[current_filter]), cv2.IMREAD_UNCHANGED)
            print(f"Filtro {current_filter} activado.")
        elif key == ord('w'):
            current_mode = 'w'
            print("Modo blanco y negro activado.")
        elif key == ord('b'):
            current_mode = 'b'
            print("Modo beige activado.")
        elif key == ord('n'):
            current_mode = 'n'
            print("Modo normal activado.")
        elif key == ord(' '):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(output_folder, f"foto_{timestamp}.png")
            cv2.imwrite(filename, frame)
            print(f"Foto guardada: {filename}")
        elif key == ord('q'):
            current_filter = None
            print("Todos los filtros desactivados.")
        elif key == ord('e'):
            print("Saliendo de la aplicación...")
            break

    picam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
