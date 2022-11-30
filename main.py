import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D
from keras.layers.convolutional import MaxPooling2D
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from datetime import datetime
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

classes = ['T-Shirt', 'Hose', 'Pullover', 'Kleid', 'Mantel',
           'Schuhe', 'Hemd', 'Turnschuhe', 'Tasche', 'Stiefel']

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

x_train = x_train / 255
x_test = x_test / 255

y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
print(model.summary())

start_time = datetime.now()

model.fit(x_train,
          y_train,
          batch_size=30,
          epochs=10,
          validation_split=0.2,
          verbose=1)

end_time = datetime.now()
time = end_time - start_time
print("Startzeit: ", start_time)
print("Endzeit: ", end_time)
print("Dauer: ", time)

predictions = model.predict(x_train)
scores = model.evaluate(x_test, y_test, verbose=1)
print("Anteil der richtigen Antworten, %: ", round(scores[1] * 100, 4))

print(predictions)

f = files.upload()

