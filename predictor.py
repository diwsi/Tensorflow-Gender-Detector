import cv2
import tensorflow as tf
import numpy as np
import os

#load trained model
model = tf.keras.models.load_model('GDM.h5')

rootPath = "assets/testImage/"
for r, d, f in os.walk(rootPath):
    for file in f:
        factImg = cv2.imread("assets/testImage/"+file)
        img = cv2.resize(factImg, (128, 128))
        #Normalise and reshape data as model input
        modelData = np.array(img/255).reshape([-1, 128, 128, 3])
        resu = model.predict(modelData)
        #Print result
        print(file+" Woman:" + str(int((100*resu[0][0]))) +
              "% Man:"+str(int((100*resu[0][1])))+"%")
