import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from numpy import array
import pickle
import numpy as np

def AiLearning():
    X = array(pickle.load(open("X.pickle", "rb")))

    y = pickle.load(open("y.pickle", "rb"))


    model = Sequential(
        [
            Flatten(input_shape=(50,50)),
            Dense(250, activation='relu'),
        ]
    )
    model.add(Dense(250, activation='relu'))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(250, activation='relu'))

    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # 10,000 epochs get it to work well
    model.fit(X,y, batch_size=25, epochs=100)

    #
    # --------------------------------------------------
    #

    testX = array(pickle.load(open("testX.pickle", "rb")))

    testy = pickle.load(open("testy.pickle", "rb"))
    testy = str(testy)
    if "0" in testy:
        testy = "compost"
    if "1" in testy:
        testy = "rcycle"
    if "2" in testy:
        testy = "trash"

    prediction = model.predict(testX)
    prediction = str(prediction)

    prediction = prediction.replace("[", "")
    prediction = prediction.replace("]", "")

    splitOne, splitTwo = prediction.split(" ", 1)
    splitThree, splitTwo = splitTwo.split(" ", 1)
    # splitTwo, splitFour = splitTwo.split(" ", 1)

    print()

    # 0 is compost and 1 is rcycyle and 2 is trash
    print()
    print("____________________________________________________________________________________________________________")
    print()
    print("Welcome To Smart Waste Collection")
    print("I am Your Ai For Today ")
    print("These Are My Predictions")
    print(prediction)
    print("Compost      Recycle        Trash")
    print()
    print()
    print("My Best Guess is ",max(splitOne,splitTwo,splitThree))
    print("Actual Value is: ", testy)
    print()
    print("Thank You for Using Smart Waste Collection and Goodbye")
    print()
    print("____________________________________________________________________________________________________________")
    print()
    print()
    # 75f talk to them
    # maybe smart things as well.

AiLearning()
