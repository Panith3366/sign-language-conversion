import numpy as np
import cv2
import os
import json
from tensorflow.keras.models import model_from_json
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load saved model
with open("Models/model_new.json", "r") as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
model.load_weights("Models/model_new.h5")
print("Loaded model from disk")

# Load and preprocess test images
def load_test_images(directory):
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

# Load test data
X_test, y_test = load_test_images('dataSet/testingData/')
X_test = X_test / 255.0  # Normalize

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(15,15))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('confusion_matrix.png')
print("Saved confusion matrix to confusion_matrix.png")

# Display sample predictions
classes = ['0'] + list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
for i in range(5):  # Show first 5 predictions
    print(f"Sample {i+1}:")
    print(f"Predicted: {classes[y_pred_classes[i]]}")
    print(f"Actual: {classes[y_test[i]]}")
    print("---")
