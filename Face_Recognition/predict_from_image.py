from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np

# Load your composite model
composite_model = keras.models.load_model("C:\\Users\\91983\\Desktop\\4th_Sem\\Mini_Project\\MINI_PROJECT\\combined_model.keras")
# Now you can use the composite model for predictions:
image_path = "C:\\Users\\91983\\Desktop\\FaceDataset\\Tejas_Pawar\\tejas_neutral2168.jpg"
img_size = 48
img = image.load_img(image_path, target_size=(img_size, img_size))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)  # Add a batch dimension
predictions = composite_model.predict(img)
# Assuming you have lists of person names and emotion labels
person_names = ['Al Faiz Ali', 'Aryan Yadav', 'Ranveer Kumar', 'Riktom Chai', 'Tejas Pawar', 'Divyanshu Kulhadiya', 'Jayanta Paul', 'Sai Dushwanth','Unknown']
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Assuming 'predictions' contains the output of your composite model
face_probabilities, emotion_probabilities = predictions

# Find the indices with the highest probabilities for both face and emotion
max_face_index = np.argmax(face_probabilities)
max_emotion_index = np.argmax(emotion_probabilities)

# Get the corresponding person name and emotion label
max_face_name = person_names[max_face_index]
max_emotion_label = emotion_labels[max_emotion_index]

# Print the results
print(f"Predicted person: {max_face_name} with probability {face_probabilities[0][max_face_index]}")
print(f"Predicted emotion: {max_emotion_label} with probability {emotion_probabilities[0][max_emotion_index]}")


# The 'predictions' will contain two sets of outputs, one for face recognition and one for emotion recognition.
