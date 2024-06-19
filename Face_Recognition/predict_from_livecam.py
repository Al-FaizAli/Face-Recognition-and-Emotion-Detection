import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load the trained face recognition model
model = keras.models.load_model('face_recognition_resnet.keras')

# Load a pre-trained face detection classifier (e.g., Haar Cascade)
# Create a face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the names of the people
# names = ['unknown', 'TEJAS', 'sai_dushwanth', 'Riktom', 'Ranveer', 'jayanta', 'divyanshu', 'Aryan', 'Al-Faiz_Ali']
names = ['Al-Faiz_Ali', 'Aryan', 'Ranveer', 'Riktom', 'TEJAS', 'divyanshu', 'jayanta', 'sai_dushwanth', 'unknown']

# Start capturing video from your webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the detected face
        face = frame[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48))  # Resize the face to the input size of your model
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)  # Add a batch dimension

        # Use the trained model to classify the detected face
        predictions = model.predict(face)
        predicted_class = np.argmax(predictions)

        # Get the person's name based on the predicted class
        person_name = names[predicted_class]

        # Draw a rectangle around the detected face and display the person's name
        label = person_name
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with face detection and recognition
    cv2.imshow('Real-Time Face Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
