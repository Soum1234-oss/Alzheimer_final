import cv2
import numpy as np
import joblib

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

# load trained models
pca = joblib.load("pca.pkl")
svm = joblib.load("svm_model.pkl")

# load VGG16 feature extractor
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
model = Model(inputs=base_model.input, outputs=base_model.output)

# class names
classes = ["Alzheimer", "MCI", "Normal"]

# load test image
image = cv2.imread(r"C:\Mini-Project\Alzheimer_Early_Detection\dataset_combined\Normal\NoImpairment (81).jpg")

if image is None:
    print("Image not found. Check file path.")
    exit()

image = cv2.resize(image,(224,224))

image = np.expand_dims(image, axis=0)
image = preprocess_input(image)

# extract features
features = model.predict(image)
features = features.flatten().reshape(1,-1)

# apply PCA
features = pca.transform(features)

# prediction
pred = svm.predict(features)[0]
prob = svm.predict_proba(features)[0]

print("Prediction:", classes[pred])
print("Confidence:", prob[pred]*100, "%")