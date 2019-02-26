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

# 20 per validation set e i rimanenti per training)
(x_train, x_valid) = x_train[10:], x_train[:10] 
(y_train, y_valid) = y_train[10:], y_train[:10]

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


#Costruzione layer CNN

model = tf.keras.Sequential()

# Il primo strato e un operatore di convolution con filtro 2x2, cioè la 
# dimensione della finestra di convoluzione.
# L'output consiste in 64 filtri. 
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(200,200,3))) 

# Il secondo strato effettua un max pooling che riduce la dimensionalità 
# dello spazio delle feature (perciò il tempo di addestramento, e il numero di 
# parametri). La pooling window è 2x2
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

# Con la tecnica di regolarizzaione del Dropout cerco di limitare l'overfitting, 
# Ad ogni iterazione il 30% dei nodi viene "disabilitato".
model.add(tf.keras.layers.Dropout(0.3))

# Il secondo strato e un operatore di convolution con filtro 2x2, cioè la 
# dimensione della finestra di convoluzione.
# L'output consiste in 32 filtri. 
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))

model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

# L'output è l'input di una sotto-rete "densa" o fully connected, con 2 layers.
# L'input alla sotto-rete corrisponde a vettori multidimensionali, perciò prima 
# rendo l'input 1D. 
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))

# L'output della sottorete viene passato a una softmax per la classificazione.
model.add(tf.keras.layers.Dense(9, activation='softmax'))
model.summary()

# Impieghiamo una variazione dello SGD chiamata Adam, molto popolare nel DL.
# Combina i benefici di altri approcci (es. AdaGrad e RMSProp).
# Per approfondimenti: 
# Adam: A Method for Stochastic Optimization https://arxiv.org/abs/1412.6980v8

model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath='standard_CNN_color.model.weights.best.hdf5', verbose = 1, save_best_only=True)

# Col parametro validation_data specifichiamo i dati usati per la validazione
model.fit(x_train,
         y_train,
         batch_size=32,
         epochs=15,
         validation_data=(x_valid, y_valid),
         callbacks=[checkpointer])

# Carichiamo il modello con l'accuratezza migliore
model.load_weights('standard_CNN_color.model.weights.best.hdf5')

# Valutiamolo sul test set
score = model.evaluate(x_test, y_test, verbose=0)
print('\n', 'Loss accuracy:', score[0])
print('\n', 'Test accuracy:', score[1])
