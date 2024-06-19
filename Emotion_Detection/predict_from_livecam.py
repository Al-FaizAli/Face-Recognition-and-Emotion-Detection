import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

class FacialExpressionModel(object):
    EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    def __init__(self, model_json_file, model_weights_file):
        # Load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)
        # Load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.compile()

    def predict_emotion(self, img):
        # Ensure that img has the shape (None, 48, 48, 3) and dtype float32
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel image
        img = cv2.resize(img, (48, 48))
        img = np.expand_dims(img, axis=0).astype('float32')  # Add batch dimension
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]

# Create an instance of FacialExpressionModel
model = FacialExpressionModel("model.json", "emotion_recognition_resnet.keras")

# Create a face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a VideoCapture object
video_capture = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    # Read a frame from the video capture
    ret, frame = video_capture.read()

    if not ret:
        raise Exception("Could not read frame from video capture.")

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = gray_frame[y:y+h, x:x+w]

        emotion_label = model.predict_emotion(face_roi)
        emotion_label = "Emotion: " + emotion_label

        # Draw emotion label and bounding box around the face
        cv2.putText(frame, emotion_label, (x, y), font, 1, (255, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Video', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
