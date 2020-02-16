
#===============================================deeplearning code=====================================================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout,Flatten,Conv2D,MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time
import pickle
import numpy as np

NAME = "cats-vs-dogs v {}".format(int(time.time()))
                       
tensorboard = TensorBoard(log_dir='.\log\{}'.format(NAME)) 

X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))


# Normalizing

X = X/255.0
model = Sequential()

model.add(Conv2D(16,(3,3),input_shape=(100,100,1),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
# model.add(Dense(200))
# model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation("sigmoid"))



model.summary()

model.compile(loss="binary_crossentropy",
             optimizer="adam",
             metrics=["accuracy"] )


#training the dataset

model.fit(X,y,batch_size=4,epochs=10,validation_split=0.1,callbacks=[tensorboard])

