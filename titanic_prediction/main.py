import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

dataset = pd.read_csv("titanic.csv")
dataset.head()

columns = ["Name", "Ticket", "Fare", "Cabin", "PassengerId"] #Unimportant columns
dataset.drop(columns=columns, inplace=True)
dataset.head()

# Standardize Age (mean = 0, std = 1)
dataset["Age"].fillna(dataset["Age"].mean(), inplace=True)
dataset["Age"] = (dataset["Age"] - dataset["Age"].mean()) / dataset["Age"].std()
dataset.head()


#fill the empty Embarked column poits with the most common value in the column
dataset["Embarked"].fillna(dataset["Embarked"].mode()[0], inplace=True)
dataset.head()

sex = {
    "male" : 1,
    "female" : 0
}
embarked = {
    "S" : 0,
    "C" : 1,
    "Q" : 2
}

dataset["Sex"] = dataset["Sex"].map(sex) #Map Sex into digits.
dataset["Embarked"] = dataset["Embarked"].map(embarked)  #Map Embarked too into digits.
dataset.head()

target = dataset.pop("Survived") #Pop out the target from the dataset


#Build a Sequential Dense Model with "relu" activation and "sigmoid" in the last layer since our data is btw 0 or 1(Survived or not-Survived)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu", input_shape=(6,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

#Compile the model with the optimizer, loss function and accuracy metrics
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

#Splitting of data into Training and Testing set
x_train, x_test, y_train, y_test = train_test_split(dataset, target, random_state=42, test_size=.25)

#Early stop: to stop the model(epochs) on the accuracy is high enough
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


#fitting the model with the training data
model.fit(
    x_train, y_train,
    epochs=40,
    batch_size=32,
    validation_split=0.20
)


#Training Evaluation Scores
train_loss, train_acc = model.evaluate(x_train, y_train)
print(f"Train Acc: {train_acc * 100:.2f}%\nTrain Loss: {train_loss * 100:.2f}%")


#Testing Evaluation Scores
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Acc: {test_acc * 100:.2f}%\nTest Loss: {test_loss * 100:.2f}%")