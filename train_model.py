import numpy as np
import joblib

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# load features
X = np.load("features.npy")
y = np.load("labels.npy")

# convert labels to numbers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Applying PCA...")

pca = PCA(n_components=100)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

print("Training SVM...")

svm = SVC(kernel="rbf", probability=True)
svm.fit(X_train, y_train)

pred = svm.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

# save models
joblib.dump(pca, "pca.pkl")
joblib.dump(svm, "svm_model.pkl")

print("Model saved")