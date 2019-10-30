import numpy as np
import os
import cv2
import random
import pickle

DATADIR = "C:\RcycleTraining"
CATEGORIES = ["compost","rcycle","trash"]

for category in CATEGORIES:
    path = os.path.join(DATADIR,category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
        break
    break

training_data = []

def create_training_data():
    for category in CATEGORIES:

        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
                IMG_SIZE = 50
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                new_array = new_array/255
                training_data.append([new_array, class_num])
            except Exception as e:
                pass


    random.shuffle(training_data)

    for features, label in training_data:
        X = []
        y = []

        X.append(features)
        y.append(label)

        #-------------------------------------------------
        X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE)

        pickle_out = open("X.pickle", "wb")
        pickle.dump(X, pickle_out)
        pickle_out.close()

        pickle_out = open("y.pickle", "wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()

    print("Current length of training data", len(training_data))
create_training_data()

print("Ai Storage Unite has been stored || Rev2 ||")
