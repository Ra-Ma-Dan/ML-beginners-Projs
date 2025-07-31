import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import cifar10

from tensorflow.keras.datasets import fashion_mnist

(train_data, train_target), (test_data, test_target) = fashion_mnist.load_data()
print(train_data[0])

import matplotlib.pyplot as pyp
pyp.imshow(train_data[0], cmap="gray")
pyp.axis("off")
pyp.show()


# First Model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(
    metrics=["accuracy"],
    optimizer="adam",
    loss="sparse_categorical_crossentropy"
)
train_data = train_data / 255.0
test_data = test_data / 255.0

history_1 = model.fit(
    train_data, train_target,
    verbose=1, epochs=20,
    validation_split=.25,
    batch_size=32
)

train_loss, train_acc = model.evaluate(train_data, train_target)
test_loss, test_acc = model.evaluate(test_data, test_target)



# Secound Model
model_2 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation="softmax")
])
model_2.compile(
    metrics=["accuracy"],
    optimizer="adam",
    loss="sparse_categorical_crossentropy"
)

history_2 = model_2.fit(
    train_data, train_target,
    verbose=1, epochs=25,
    validation_split=.25,
    batch_size=32
)
history_2

train_loss, train_acc = model_2.evaluate(train_data, train_target)
test_loss, test_acc = model_2.evaluate(test_data, test_target)



# Third Model
model_3 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation="softmax")
])
model_3.compile(
    metrics=["accuracy"],
    optimizer="adam",
    loss="sparse_categorical_crossentropy"
)

history_3 = model_3.fit(
    train_data, train_target,
    verbose=1, epochs=20,
    validation_split=.25,
    batch_size=32
)
history_3

train_loss, train_acc = model_3.evaluate(train_data, train_target)
test_loss, test_acc = model_3.evaluate(test_data, test_target)

def pie_plot_models(agent, angle):
    tr_loss, tr_accu = agent.evaluate(train_data, train_target)
    te_loss, te_accu = agent.evaluate(test_data, test_target)
    labels = ["Train Accuracy", "Train Loss", "Test Accuracy", "Test Loss"]
    values = [tr_accu, tr_loss, te_accu, te_loss]

    total = sum(values)
    percentages = [(value / total) * 100 for value in values]
    
    plt.figure(figsize=(6,6))
    plt.pie(percentages, labels=labels, autopct="%1.1f%%", startangle=angle)
    plt.title(f"Overall Metrics for {agent}: (both training and testing accuracies and losses)")
    plt.show()

def bar_plot_models(agent):
    tr_loss, tr_accu = agent.evaluate(train_data, train_target)
    te_loss, te_accu = agent.evaluate(test_data, test_target)
    labels = ["Train Accuracy", "Train Loss", "Test Accuracy", "Test Loss"]
    values = [tr_accu, tr_loss, te_accu, te_loss]
    
    plt.figure(figsize=(8,5))
    plt.bar(labels, values, color=['green', 'blue', 'orange', 'red'])
    plt.title('Final Metrics')
    plt.ylabel('Value')
    plt.ylim(0, 1.0)
    plt.show()
