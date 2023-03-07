
Hello! this is Aman Kumar.

This project made by me is a AI Handwritten Digits Recognition project where user draws the digits and AI predicts the digit.

Before implementation of this project. Certain libraries and packages i have installed.
1. Keras
2. Numpy
3. Matplotlib
4. Tensorflow
5. Mnist
6. Pillow
7. Tkinter

Above packages can be installed using cmd in windows or terminal in mac or shell in Linux.
Be sure to intsall all the above libraries or packages.

There are 2 types of python files I have attached in 2 different branches. One is the python code for training the data, other for executing the python code output.

First take the training data python code and run it. It will take few minutes to train for certain level of accuracy. Then run the output executed python code. It will execute the output.

Limits:
There may be chances that code might not work properly. Suppose if you draw 7 it tells you 8, this is due to AI recognization capacity but rest of the time it will work properly.


# Python Code for training the data. Be sure to install the necessary libraries or packages previously.

import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as k

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train /= 255
x_test /= 255

batch_size = 128
num_classes = 10
epochs = 100

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), activation="relu", input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation="softmax"))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=["accuracy"])
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print("loss:", score[0])
print("accuracy:", score[1])

model.save("mnist.h5")
 
