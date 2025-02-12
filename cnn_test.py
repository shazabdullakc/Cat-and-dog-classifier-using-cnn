import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

# Set up data generators
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

# Load and preprocess training data
train_generator = train_datagen.flow_from_directory(
    "C:\\MyFiles\\Project\\AI camera\\dataset\dataset\\training_set",
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Load and preprocess validation data
validation_generator = test_datagen.flow_from_directory(
    "C:\\MyFiles\\Project\\AI camera\\dataset\\dataset\\test_set",
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Example of how to use the generators in a model
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x = train_generator, validation_data = validation_generator, epochs = 25)

test_image = image.load_img("C:\MyFiles\Project\AI camera\dataset\dataset\single_prediction\cat_or_dog_2.jpg", target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
train_generator.class_indices
if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat'
print(prediction)