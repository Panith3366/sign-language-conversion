import numpy as np
import cv2
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split

# Load and preprocess images
def load_images(directory):
    images = []
    labels = []
    classes = ['0'] + list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    
    for idx, cls in enumerate(classes):
        path = os.path.join(directory, cls)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128))
            img = np.array(img).reshape(128, 128, 1)
            images.append(img)
            labels.append(idx)
    
    return np.array(images), np.array(labels)

# Load training data
X, y = load_images('dataSet/trainingData/')
X = X / 255.0  # Normalize
y = to_categorical(y)  # One-hot encode

# Split into train and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Create CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(27, activation='softmax'))  # 26 letters + blank

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train,
          batch_size=32,
          epochs=10,
          verbose=1,
          validation_data=(X_val, y_val))

# Save model
model_json = model.to_json()
with open("Models/model_new.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("Models/model_new.h5")
print("Saved model to disk")
