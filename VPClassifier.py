import os
import numpy as np
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Prepare data
train_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    'E:/proj/T',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    'E:/TT',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# Save the model
model.save('vehicle_plate_classifier.h5')

# Load saved model for classification
from tensorflow.keras.models import load_model
model = load_model('vehicle_plate_classifier.h5')

# Classify and sort unsorted images
unsorted_folder = 'C:/BIGDATA'
sorted_yellow_folder = 'C:/P/1Y_P'
sorted_white_folder = 'C:/P/1W_P'

# Create sorted folders if they don't exist
if not os.path.exists(sorted_yellow_folder):
    os.makedirs(sorted_yellow_folder)

if not os.path.exists(sorted_white_folder):
    os.makedirs(sorted_white_folder)

for filename in os.listdir(unsorted_folder):
    img_path = os.path.join(unsorted_folder, filename)
    img = load_img(img_path, target_size=(150, 150))
    img_tensor = img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.0

    prediction = model.predict(img_tensor)

    if prediction < 0.5:
        print(f"{filename} is a private vehicle with a white plate.")
        shutil.move(img_path, os.path.join(sorted_white_folder, filename))
    else:
        print(f"{filename} is a commercial vehicle with a yellow plate.")
        shutil.move(img_path, os.path.join(sorted_yellow_folder, filename))
