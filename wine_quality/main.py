import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine

data = load_wine()

dataset = pd.DataFrame(data=data.data, columns=data.feature_names)
dataset["target"] = data.target
columns = data.feature_names
target_names = data.target_names

target = dataset.pop("target")
x_train, x_test, y_train, y_test = train_test_split(dataset, target, test_size=.25, random_state=42)

x_train.head()

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu", input_shape=(13,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])

history = model.fit(
    x_train, y_train,
    epochs=40,
    verbose=1,
    batch_size=32,
    validation_split=.15
)

train_loss, train_acc = model.evaluate(x_train, y_train)
print(f"Train Accuracy: {train_acc * 100:.2f}% and Train Loss: {train_loss * 100:.2f}%")

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}% and Test Loss: {test_loss * 100:.2f}%")


# Testing  Random Forest Classifier from Scikit-Learn
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
