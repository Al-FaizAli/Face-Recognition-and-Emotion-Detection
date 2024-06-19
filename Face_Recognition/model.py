import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from livelossplot.inputs.tf_keras import PlotLossesCallback

# Define your data directory and image size
data_dir = "C:\\Users\\91983\\Downloads\\face_data_2023_mini_project\\face_data_2023_mini_project\\face_recognitation"
img_size = 48
batch_size = 16

# Load your custom dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_size, img_size),
    batch_size=batch_size,
    shuffle=True,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_size, img_size),
    batch_size=batch_size,
    shuffle=True,
)

n1 = len(train_ds)
n2 = len(val_ds)
class_names = train_ds.class_names
print(class_names)
num_classes = len(os.listdir(data_dir))  # The number of people you want to recognize

# One-hot encode the labels
train_ds = train_ds.map(lambda x, y: (x, tf.one_hot(y, depth=9)))
val_ds = val_ds.map(lambda x, y: (x, tf.one_hot(y, depth=9)))

# Build the model
base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(img_size, img_size, 3),
    pooling='avg'
)

x = layers.Dense(512, activation='relu')(base_model.output)
x = layers.Dropout(0.5)(x)
x = layers.Dense(num_classes, activation='softmax')(x)

model = keras.models.Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training parameters
epochs = 15
steps_per_epoch = n1 // batch_size
validation_steps = n2 // batch_size

# Define callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.00001, mode='auto')
checkpoint = ModelCheckpoint("face_recognition_resnet.keras", monitor='val_accuracy', save_weights_only=True, mode='max', verbose=1)
callbacks = [PlotLossesCallback(), checkpoint, reduce_lr]
callbacks = [checkpoint, reduce_lr]

# Train the model
history = model.fit(
    x=train_ds,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=val_ds,
    validation_steps=validation_steps,
    callbacks=callbacks
)

# Save the model
model.save('face_recognition_resnet.keras')
model.save_weights('face_recognition_resnet_weights.h5')
