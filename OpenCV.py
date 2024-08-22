import cv2
import time
import numpy as np
import dlib

age_net = cv2.dnn.readNet("models/age_net.caffemodel", "models/age_deploy.prototxt")
gender_net = cv2.dnn.readNet("models/gender_net.caffemodel", "models/gender_deploy.prototxt")
frontal_face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

AGE_LABELS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LABELS = ['Homem', 'Mulher']


def calculate_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def is_looking_at_camera(left_eye, right_eye):
    left_ear = calculate_ear(left_eye)
    right_ear = calculate_ear(right_eye)
    threshold = 0.33
    average_ear = (left_ear + right_ear) / 2
    return average_ear > threshold


def main():
    cap = cv2.VideoCapture(1)

    start_time = time.time()
    last_prediction = ""
    last_looking_status = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = frontal_face_detector(gray)

        if not faces:
            cv2.putText(frame, "Rosto nao detectado", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                        cv2.LINE_AA)
        else:
            faces = sorted(faces, key=lambda face: (face.right() - face.left()) * (face.bottom() - face.top()),
                           reverse=True)
            face = faces[0]
            h, w = frame.shape[:2]

            x1 = max(0, face.left() - 10)
            y1 = max(0, face.top() - 10)
            x2 = min(w, face.right() + 10)
            y2 = min(h, face.bottom() + 10)

            face_region = frame[y1:y2, x1:x2]
            face_region = cv2.resize(face_region, (227, 227))
            face_region = cv2.equalizeHist(cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY))
            face_region = cv2.cvtColor(face_region, cv2.COLOR_GRAY2BGR)

            shape = shape_predictor(gray, face)
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            looking_at_camera = is_looking_at_camera(left_eye, right_eye)
            last_looking_status = f"Olhando para Camera" if looking_at_camera else f"Nao Olhando para Camera"
            elapsed_time = time.time() - start_time

            if elapsed_time >= 5 and looking_at_camera:
                start_time = time.time()

                blob = cv2.dnn.blobFromImage(face_region, 1, (227, 227), (78.4263377603, 87.7689143744, 114.895847746),
                                             swapRB=False)

                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender = GENDER_LABELS[gender_preds[0].argmax()]

                age_net.setInput(blob)
                age_preds = age_net.forward()
                age = AGE_LABELS[age_preds[0].argmax()]

                last_prediction = f"Genero: {gender}, Idade: {age}"

                print(last_prediction)

            cv2.putText(frame, last_prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            color = (0, 255, 0) if looking_at_camera else (0, 0, 255)
            cv2.putText(frame, last_looking_status, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        cv2.imshow("Age and Gender Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
