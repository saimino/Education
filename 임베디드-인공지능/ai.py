import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt


img_row = 28
img_col = 28

batch_size = 128
num_classes = 10
epochs = 5


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

input_shape = (img_row, img_col, 1)
x_train = x_train.reshape(x_train.shape[0], img_row, img_col, 1)
x_test = x_test.reshape(x_test.shape[0], img_row, img_col, 1)

x_train = x_train.astype('float32') / 255.
y_test = y_test.astype('float32') / 255.


y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = keras.Sequential()


model.add(keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same', input_shape=input_shape))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


model.add(keras.layers.Conv2D(64, (2, 2), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))


model.add(keras.layers.Dropout(0.25))


model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(1000, activation='relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(num_classes, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))



model.summary()


plt.imshow(x_test[0].reshape(img_row, img_col), cmap='Greys', interpolation='nearest')
plt.show()

 


model.evaluate(x_test, y_test)



model.save('My_MNIST_CNN_DATA.h5')
