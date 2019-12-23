import time
import numpy as np
import cv2
import os
import random
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

trainData = []
trainLabel = []

for subdir, dirs, files in os.walk("./assets/mwc/Refined/"):
    #shuffle  array for better training
    random.shuffle(files)
    for file in files:
        filepath = subdir + os.sep + file
        img = cv2.imread(filepath)
        trainData.append(img/255)#normalisation
        trainLabel.append(1 if file.startswith('men') else 0)
        print(file)

#reshape array to numpy format
trainData = np.array(trainData).reshape([-1, 128, 128, 3])
trainLabel = np.array(trainLabel)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3),
                        activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3),  activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3),  activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3),  activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

#logs for tensorboard
tbLog = tf.keras.callbacks.TensorBoard(
    log_dir="logs\\gdmL{}".format(time.time()),
    histogram_freq=1)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(trainData, trainLabel, epochs=50, callbacks=[tbLog])
model.save('GDM.h5')
