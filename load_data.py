import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

TRAINDIR = "data/training"
TESTDIR = "data/testing"

CATEGORIES = ["Black","Blue","Brown","Green",
              "Orange","Red","Violet","White","Yellow"]

for category in CATEGORIES:  # do classes
    path = os.path.join(TRAINDIR,category)  # create path to classes
    for img in os.listdir(path):  # iterate over each image per classes
        img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_COLOR)
        plt.imshow(img_array)
        plt.show()
        break
    break 
print (img_array.shape)

#decidiamo di fare il resize di tutte le immagini a 200x200
IMG_WIDTH=200
IMG_HEIGHT=200

#create training set
training_data = []

def create_training_data():
    for category in CATEGORIES:  # do classes

        path = os.path.join(TRAINDIR,category)  # create path to classes
        class_num = CATEGORIES.index(category)  # get the classification  (0 to 14). 0=Adenovirus, ..., 14=WestNile

        for img in tqdm(os.listdir(path)):  # iterate over each image per class
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_COLOR)  # convert to array
                new_array = cv2.resize(img_array, (IMG_WIDTH,IMG_HEIGHT))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_training_data()
print(len(training_data))

#shuffle the data
import random

random.shuffle(training_data)

#make our model
x_train = []
y_train = []

for features,label in training_data:
    x_train.append(features)
    y_train.append(label)

x_train = np.array(x_train).reshape(-1, IMG_WIDTH,IMG_HEIGHT,3)
y_train= np.array(y_train)
print(x_train.shape)
print(y_train.shape)

#create test set
test_data = []

def create_test_data():
    for category in CATEGORIES:  # do classes

        path = os.path.join(TESTDIR,category)  # create path to classes
        class_num = CATEGORIES.index(category)  # get the classification  (0 to 14). 0=Adenovirus, ..., 14=WestNile

        for img in tqdm(os.listdir(path)):  # iterate over each image per class
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_COLOR)  # convert to array
                new_array = cv2.resize(img_array, (IMG_WIDTH,IMG_HEIGHT))  # resize to normalize data size
                test_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_test_data()
print(len(test_data))

#shuffle the data
import random

random.shuffle(test_data)

#make our model
x_test = []
y_test = []

for features,label in test_data:
    x_test.append(features)
    y_test.append(label)

x_test = np.array(x_test).reshape(-1, IMG_WIDTH,IMG_HEIGHT,3)
y_test= np.array(y_test)
print(x_test.shape)
print(y_test.shape)

print (x_train.shape,len(y_train))
print (x_test.shape, len(y_test))

import tensorflow as tf
import matplotlib.pyplot as plt

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

print("Number of train data - " + str(len(x_train)))
print("Number of test data - " + str(len(x_test)))

#Let's save this data, so that we don't need to keep calculating
#it every time we want to play with the neural network model
import pickle

pickle_out = open("x_train.pickle","wb")
pickle.dump(x_train, pickle_out)
pickle_out.close()

pickle_out = open("y_train.pickle","wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()

pickle_out = open("x_test.pickle","wb")
pickle.dump(x_test, pickle_out)
pickle_out.close()

pickle_out = open("y_test.pickle","wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()
