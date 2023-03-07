
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

There are 2 types of python codes i have mentioned below. One is the python code for training the data, other for executing the python code output.

First take the training data python code and run it. It will take few minutes to train for certain level of accuracy. Then run the output executed python code. It will execute the output.

Limits:
There may be chances that code might not work properly. Suppose if you draw 7 it tells you 8, this is due to AI recognization capacity but rest of the time it will work properly.

Create a python file. Copy-paste my below code from line 1 to line 85. save it. Then create another python file copy-paste my below code from line 90 to line 157 and save it. Then run the first python file wait for few minutes then run the next code. Be sure to install the necessary libraries or packages previously before running it.

# Python Code for training the data. 

# Line 1

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

# Line 85
 
 
# Python code for executing the output.

# Line 90

from keras.models import load_model                    

from tkinter import *

import tkinter as tk

import win32gui

from PIL import ImageGrab, Image

import numpy as np

model=load_model('mnist.h5')

def predict_digit(img):
    img=img.resize((28,28))
    img=img.convert('L')
    img=np.array(img)
    img=img.reshape(1,28,28,1)
    img=img/255.0
    
    res=model.predict([img])[0]
    return np.argmax(res),max(res)

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        
        self.x = self.y = 0
        
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self,text = "Draw..",font=("Helvetica",48))
        self.classify_btn = tk.Button(self, text = "Recognise", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text = "Clear", command=self.clear_all)
        
        self.canvas.grid(row=0,column=0,pady=2,sticky=W,)
        self.label.grid(row=0,column=1,pady=2,padx=2)
        self.classify_btn.grid(row=1,column=1,pady=2,padx=2)
        self.button_clear.grid(row=1,column=0,pady=2)
        
        self.canvas.bind("<B1-Motion>",self.draw_lines)
        
    def clear_all(self):
        self.canvas.delete("all")
        
    def classify_handwriting(self):
        HWND=self.canvas.winfo_id()
        rect=win32gui.GetWindowRect(HWND)
        a,b,c,d = rect
        rect=(a+4,b+4,c-4,d-4)
        im=ImageGrab.grab(rect)
        
        digit,acc=predict_digit(im)
        self.label.configure(text=str(digit)+', '+str(int(acc*100))+'%')
        
    def draw_lines(self, event):
        self.x=event.x
        self.y=event.y
        r=8
        self.canvas.create_oval(self.x-r,self.y-r,self.x+r,self.y+r,fill='black')
                                
app = App()

mainloop()                   

# Line 157

