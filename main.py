__author__ = 'Matthew'

import pandas as pd
import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, Conv3D

img_rows, img_cols, color_cnt = 64, 64, 1
num_classes = 2
num_images = 25000
epochs = 10
batch_size = 100

data = pd.read_csv('dogsvscats-train-med-grey.csv')

y = keras.utils.to_categorical(data['class'], num_classes)

data['data'] = data['data'].apply(lambda x: np.array([float(x_split)/255 for x_split in x[1:-1].split(',')]))
x = np.concatenate(data['data'])
x = x.reshape(num_images, img_rows, img_cols, color_cnt)

# from PIL import Image
# img = Image.fromarray(x[3].astype('uint8'))
# img.save('check.jpg')

dogvscat_model = Sequential()
dogvscat_model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=(img_rows, img_cols, color_cnt)))
dogvscat_model.add(Conv2D(64, (5, 5), activation='relu'))
dogvscat_model.add(Conv2D(32, (5, 5), activation='relu'))
dogvscat_model.add(Conv2D(64, (5, 5), activation='relu'))
dogvscat_model.add(Conv2D(32, (5, 5), activation='relu'))
dogvscat_model.add(Conv2D(64, (5, 5), activation='relu'))
dogvscat_model.add(Flatten())
dogvscat_model.add(Dense(1024, activation='relu'))
dogvscat_model.add(Dropout(0.8))
dogvscat_model.add(Dense(num_classes, activation='softmax'))

dogvscat_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

dogvscat_model.fit(x, y,
          batch_size=batch_size,
          epochs=epochs,
          validation_split = 0.2)