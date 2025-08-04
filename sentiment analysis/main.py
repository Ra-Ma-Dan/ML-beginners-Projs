import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras import datasets
from keras.preprocessing import sequence
import os

VOCAB_SIZE = 88584 # our Vocabulary size(size of the unique words known in our dataset)
BATCH_SIZE = 64
MAX_LEN = 250 # maximum length of input to be recieved by the model

#Loading the IMDB data from KERAS
(train_data, train_target), (test_data, test_target) = datasets.imdb.load_data(num_words=VOCAB_SIZE)
train_data.shape

#Padding the data if not up to our max length and truncating it if above the max length
train_data = sequence.pad_sequences(train_data, MAX_LEN)
test_data = sequence.pad_sequences(test_data, MAX_LEN)

#Building a Sequential RNN( using LSTM) model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32, input_shape=(MAX_LEN,)), #layer for text vectorizing(embedding)
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compiling the built Model with adam optimizer
model.compile(
    optimizer="adam",
    metrics=["accuracy"],
    loss="binary_crossentropy"
)

# Fitting the model with train data
history = model.fit(
    train_data, train_target,
    epochs=10, batch_size=32,
    validation_split=.25,
    verbose=1
)


# Model 1 evaluation
loss, acc = model.evaluate(train_data, train_target)
tr_loss, tr_acc = model.evaluate(test_data, test_target)


# Model with RMSprop optimizer and 64 LSTM neurons
model_2 = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 64, input_shape=(MAX_LEN,)),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model_2.compile(
    optimizer="rmsprop",
    metrics=["accuracy"],
    loss="binary_crossentropy"
)

history_2 = model_2.fit(
    train_data, train_target,
    epochs=5, batch_size=32,
    validation_split=.25,
    verbose=1
)


# Model 2's Evaluation on train and test Data
loss, acc = model_2.evaluate(train_data, train_target)
tr_loss, tr_acc = model_2.evaluate(test_data, test_target)