import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Activation
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
from keras import regularizers
import time

# Carichiamo il training set
pickle_in = open("x_train.pickle","rb")
x_train = pickle.load(pickle_in)

pickle_in = open("y_train.pickle","rb")
y_train = pickle.load(pickle_in)

# Carichiamo il test set
pickle_in = open("x_test.pickle","rb")
x_test = pickle.load(pickle_in)

pickle_in = open("y_test.pickle","rb")
y_test = pickle.load(pickle_in)

x_train = np.array(x_train)
y_train = np.array(y_train)

print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

#per debug
color_labels = ["Black","Blue","Brown","Green","Orange",
              "Red","Violet","White","Yellow"]

# Immagine a caso
img_index = 5
label_index = y_train[img_index]
print ("y = " + str(label_index) + " " +(color_labels[label_index]))
plt.imshow(x_train[img_index].squeeze())  #squeeze(): Remove single-dimensional entries from the shape of an array.
plt.show()

# Suddividiamo i dati di training validation e test.

# 150 per validation set e i rimanenti 150 per training)
(x_train, x_valid) = x_train[20:], x_train[:20] 
(y_train, y_valid) = y_train[20:], y_train[:20]

num_classes=9
# Rappresentazione One-hot
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_valid = tf.keras.utils.to_categorical(y_valid, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

print (y_train[0])
print (y_train[2])
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

print(x_train.shape[0], 'train set')
print(x_valid.shape[0], 'validation set')
print(x_test.shape[0], 'test set')


def ConvBlock(model, layers, filters):
    '''Create [layers] layers consisting of zero padding, a convolution with [filters] 3x3 filters and batch normalization. Perform max pooling after the last layer.'''
    for i in range(layers):
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters, (2, 2), activation='relu'))
        model.add(BatchNormalization(axis=3))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

def create_model():
    '''Create the FCN and return a keras model.'''

    model = Sequential()

    # Input image: 200x200x3
    model.add(Lambda(lambda x: x, input_shape=(200, 200, 3)))
    ConvBlock(model, 1, 32)
    # 
    ConvBlock(model, 1, 64)
    # 
    ConvBlock(model, 1, 128)
    # 
    ConvBlock(model, 1, 128)
    # 
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(9, (2, 2), activation='relu'))
    model.add(GlobalAveragePooling2D())
    # 
    model.add(Activation('softmax'))
    
    return model

# Create the model and compile
model = create_model()
model.summary()

model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath='standard_CNN_color.model.weights.best.hdf5', verbose = 1, save_best_only=True)

# Col parametro validation_data specifichiamo i dati usati per la validazione
model.fit(x_train,
         y_train,
         batch_size=32,
         epochs=5,
         validation_data=(x_valid, y_valid),
         callbacks=[checkpointer])

# Carichiamo il modello con l'accuratezza migliore
model.load_weights('standard_CNN_color.model.weights.best.hdf5')

# Valutiamolo sul test set
score = model.evaluate(x_test, y_test, verbose=0)
print('\n', 'Loss accuracy:', score[0])
print('\n', 'Test accuracy:', score[1])









