#Hande the Imports
import pandas as pd #for data manipulation
import tensorflow as tf #for the main MODEL
from sklearn.model_selection import train_test_split

COLUMNS = ["SeplaLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"] #columns in our dataset
data = pd.read_csv("../iris_data.csv", names=COLUMNS, header=0) #read the Iris dataset as pd dataframe
data.head() #display the first five rows of the data

mapping_data = {
    'setosa' : 0, 
    'versicolor' : 1, 
    'virginica' : 2
}
data["Species"] = data["Species"].map(mapping_data) #Since the species column is Categorical, 
data["Species"]               # the map function is used to covert all existencs of each word to a cirtain digit
                            # where 1 represent Setosa, 1 represent Vernicolor and 2 represnts Virginaca.

Y_data = data.pop("Species") #pop returns a specific column from the data and mutate the data. So the data
Y_data                       # does not contain the "Species" column anymore.


x_train, x_test, y_train, y_test = train_test_split(data, Y_data, test_size=0.25, random_state=42) #Spliting the data set into training and testing set for proper traing and testing
x_train


#Now Building the Layers of the Model, First layer(16 neurons, relu activation and telling the model it should expect 4 inputs(as in our dataset))
#Second Layer is another 16 neurons with rlu activation again, but this time our model is already aware of the expected inputs.
#The Third layer is the Out configurer that tells the model our output can only br between the three species of FLOWERS.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu", input_shape=(4,)), 
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])

#Then Model Complation after the Building
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)


#Then fitting the model with training dataset(inputs and expected outputs)
#bacth_size expresses the batch sizing, epochs: number of time the model shuold see the same data over and over
#validation slip: train model with 80% of data and validate it with 20%
#verbose: show the process
model.fit(
    x_train, y_train,
    batch_size=8,
    epochs=50,
    validation_split=0.2,
    verbose=1
)


train_los, train_acc = model.evaluate(x_train, y_train) #Evaluation on how much the model understand the data while it's been trained
print(f"Train Acc: {train_acc:.2f} \nTrain Loss: {train_los:.2f}")

test_loss, test_acc = model.evaluate(x_test, y_test) #Evaluation on how accurate the model is while it's been tested
print(f"Test Acc: {test_acc:.2f} \nTest Loss: {test_loss:.2f}")


#make prediction with
all_predictions = model.predict(x_test) #this return a list of prbabilties the model predicted for each species
prediction = all_predictions.argmax(axis=1) #now returns the real one with the highest propability value