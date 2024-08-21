import cv2
import numpy as np
import dlib

# TODO: Improve age prediction

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")


def load_models():
    try:
        face_net = cv2.dnn.readNet("models/opencv_face_detector_uint8.pb", "models/opencv_face_detector.pbtxt")
        age_net = cv2.dnn.readNet("models/age_net.caffemodel", "models/age_deploy.prototxt")
        gender_net = cv2.dnn.readNet("models/gender_net.caffemodel", "models/gender_deploy.prototxt")
        return face_net, age_net, gender_net
    except cv2.error as e:
        print("Error loading models:", e)
        return None, None, None


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    smoothed = cv2.GaussianBlur(equalized, (5, 5), 0)
    preprocessed = cv2.cvtColor(smoothed, cv2.COLOR_GRAY2BGR)
    return preprocessed


def detect_faces(face_net, image):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], True, False)
    face_net.setInput(blob)
    detections = face_net.forward()
    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            face_boxes.append((x1, y1, x2, y2))
    return face_boxes


def calculate_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def is_looking_at_camera(left_eye, right_eye):
    left_ear = calculate_ear(left_eye)
    right_ear = calculate_ear(right_eye)
    threshold = 0.26
    average_ear = (left_ear + right_ear) / 2
    return average_ear > threshold, average_ear


def predict_age_gender(face, age_net, gender_net, MODEL_MEAN_VALUES):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = 'Homem' if gender_preds[0][0] > gender_preds[0][1] else 'Mulher'

    age_net.setInput(blob)
    age_preds = age_net.forward()
    age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    age_confidences = age_preds[0]
    age_index = np.argmax(age_confidences)
    age_index = min(max(age_index, 0), len(age_list) - 1)
    age = age_list[age_index]
    age_confidence = age_confidences[age_index]

    return gender, age, age_confidence


def process_frame(frame, face_net, age_net, gender_net, MODEL_MEAN_VALUES):
    h, w = frame.shape[:2]
    preprocessed_frame = preprocess_image(frame)
    face_boxes = detect_faces(face_net, preprocessed_frame)

    if not face_boxes:
        return frame

    for (x1, y1, x2, y2) in face_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(h / 150)), 8)
        face = frame[y1:y2, x1:x2]
        gender, age, age_confidence = predict_age_gender(face, age_net, gender_net, MODEL_MEAN_VALUES)
        label = f"{gender}, Idade: {age} ({age_confidence * 100:.2f}%)"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        faces_dlib = detector(gray_face)
        if not faces_dlib:
            cv2.putText(frame, "Nao esta olhando para a camera",
                        (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        for dlib_face in faces_dlib:
            shape = predictor(gray_face, dlib_face)
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            for (x, y) in np.concatenate((left_eye, right_eye)):
                cv2.circle(frame, (x1 + x, y1 + y), 2, (0, 255, 0), -1)
            looking_at_camera, ear_value = is_looking_at_camera(left_eye, right_eye)

            if looking_at_camera:
                cv2.putText(frame, "Esta olhando para a camera",
                            (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Nao esta olhando para a camera",
                            (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return frame


def main():
    face_net, age_net, gender_net = load_models()
    if not face_net or not age_net or not gender_net:
        return
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    webcam = cv2.VideoCapture(0)  # Ensure the correct camera index
    if not webcam.isOpened():
        print("Error: Could not open webcam")
        return
    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Error: Failed to grab frame")
            break
        frame = process_frame(frame, face_net, age_net, gender_net, MODEL_MEAN_VALUES)
        cv2.imshow("Age-Gender Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
