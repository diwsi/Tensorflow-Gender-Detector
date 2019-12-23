import pickle
import cv2
import os


def readFile(path):
    with open(path, 'rb') as file:
        return pickle.load(file, encoding='bytes')


def fetchFiles(pref):
    for subdir, dirs, files in os.walk("./assets/mwc/"+pref+"/"):
        for file in files:
            if(file.endswith(".jpg")):
                print(pref+file)
                filepath = subdir + os.sep + file
                img = cv2.imread(filepath)
                mindim = min(img.shape[0], img.shape[1])
                y0 = int((img.shape[0]/2)-mindim/2)
                y1 = int((img.shape[0]/2)+mindim/2)
                x0 = int((img.shape[1]/2)-mindim/2)
                x1 = int((img.shape[1]/2)+mindim/2)
                crop_img = img[y0:y1, x0:x1]
                crop_img = cv2.resize(crop_img, (128, 128))
                cv2.imwrite("./assets/mwc/Refined/"+pref+file, crop_img)


fetchFiles("men")
fetchFiles("women")
 
