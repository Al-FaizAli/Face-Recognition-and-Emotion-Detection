import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from livelossplot.inputs.tf_keras import PlotLossesCallback

data_dir ="C:\\Users\\91983\\Downloads\\face_data_2023_mini_project\\face_data_2023_mini_project\\facial_emotion\\emotion"
img_size = 48
batch_size = 16

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123, 
    image_size=(img_size, img_size), 
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, 
    validation_split=0.2, 
    subset="validation", 
    seed=123, 
    image_size=(img_size, img_size), 
    batch_size=batch_size
)
n1 = len(train_ds)
n2 = len(val_ds)
class_names = train_ds.class_names
print(class_names)
num_classes = len(os.listdir(data_dir))  # The number of people you want to recognize

# One-hot encode the labels
train_ds = train_ds.map(lambda x, y: (x, tf.one_hot(y, depth=7)))
val_ds = val_ds.map(lambda x, y: (x, tf.one_hot(y, depth=7)))

base_model = tf.keras.applications.ResNet50(
    include_top=False,  # Exclude the fully connected layers
    weights='imagenet',  # Use pre-trained weights
    input_shape=(img_size, img_size, 3),  # Adjust to your input image size and channels
    pooling='avg'  # Global average pooling for output
)

# Add custom fully connected layers for emotion classification
x = layers.Dense(512, activation='relu')(base_model.output)
x = layers.Dropout(0.5)(x)
x = layers.Dense(num_classes, activation='softmax')(x)  # 7 emotion classes

model = keras.models.Model(inputs=base_model.input, outputs=x)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

epochs = 15
steps_per_epoch = n1 // batch_size
validation_steps = n2 // batch_size
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.00001, mode='auto')
checkpoint = ModelCheckpoint("emotion_recognition_resnet.keras", monitor='val_accuracy', save_weights_only=True, mode='max', verbose=1)
callbacks = [PlotLossesCallback(), checkpoint, reduce_lr]
callbacks = [checkpoint, reduce_lr]
history = model.fit(
    x=train_ds,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=val_ds,
    validation_steps=validation_steps,
    callbacks=callbacks
)

model_json = model.to_json()
model.save('emotion_recognition_resnet.keras')
with open("model.json", "w") as json_file:
    json_file.write(model_json)