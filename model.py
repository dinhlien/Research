import tensorflow as tf
import h5py as h5
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, Dropout, Reshape, Conv2D, MaxPooling2D, MaxPooling1D
import matplotlib.pyplot as plt
import hickle as hkl

file = h5.File('/Users/lelm01/Desktop/RFResearch/2018.01.OSC.0001_1024x2M.h5/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5', 'r')
X_data = file['X'][:]
Y_data = file['Y'][:]
Z_data = file['Z'][:]
file.close()

# Splitting train/test data
np.random.seed(1)
n_examples = X_data.shape[0]
n_train = int(n_examples * 0.8)
train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
test_idx = list(set(range(0, n_examples))-set(train_idx))
X_train = X_data[train_idx]
Y_train = Y_data[train_idx]
X_test = X_data[test_idx]
Y_test = Y_data[test_idx]
Z_train = Z_data[train_idx]
Z_test = Z_data[test_idx]

classes = ['32PSK',
 '16APSK',
 '32QAM',
 'FM',
 'GMSK',
 '32APSK',
 'OQPSK',
 '8ASK',
 'BPSK',
 '8PSK',
 'AM-SSB-SC',
 '4ASK',
 '16PSK',
 '64APSK',
 '128QAM',
 '128APSK',
 'AM-DSB-SC',
 'AM-SSB-WC',
 '64QAM',
 'QPSK',
 '256QAM',
 'AM-DSB-WC',
 'OOK',
 '16QAM']

X_train,X_test,Y_train,Y_test,Z_train,Z_test,classes = hkl.load('data.hkl') # load data from hickle file

dr = 0.5      # drop out rates
epoch = 10    # n. epochs
batch_size = 512

## FCNN

fcnn_mod = Sequential()
fcnn_mod.add(Flatten(input_shape=(1024, 2)))
fcnn_mod.add(Dense(256, input_shape=(1024, 2), activation='relu', name="dense1"))
fcnn_mod.add(Dropout(dr))
fcnn_mod.add(Dense(256, activation='relu', name="dense2"))
fcnn_mod.add(Dropout(dr))
fcnn_mod.add(Dense(128, activation='relu', name="dense3"))
fcnn_mod.add(Dropout(dr))
fcnn_mod.add(Dense(24, activation='softmax', name="output"))
fcnn_mod.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
fcnn_mod.summary()

# # training
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
fcnn_filepath = "fcnn01_weights.h5"  # check point filepath
fcnn_his = fcnn_mod.fit(X_train, Y_train,
                         batch_size=batch_size,
                         epochs=50,
                         validation_split=0.25,
                         verbose=2,
                         callbacks= [keras.callbacks.ModelCheckpoint(fcnn_filepath,
                                                                   monitor='val_accuracy',
                                                                   verbose=0,
                                                                   save_best_only=True,
                                                                   mode='auto'),
                                     keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                                   patience=5,
                                                                   verbose=0,
                                                                   mode='auto')
                                     ]
                        )


# re-load weights after training
fcnn_mod.load_weights(fcnn_filepath)

# predict
# preds = fcnn_mod.predict(X_test)
# print(preds)

# evaluation
fcnn_score = fcnn_mod.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print(fcnn_score)

# plot train vs test accuracy per epoch
plt.figure()
plt.plot(fcnn_his.history['loss'])
plt.plot(fcnn_his.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'])
plt.show()

plt.figure()
plt.plot(fcnn_his.history['accuracy'])
plt.plot(fcnn_his.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'])
plt.show()


## CNN
# Conv1D
cnn_mod = Sequential()
cnn_mod.add(Reshape((2048, 1), input_shape=(1024, 2)))
cnn_mod.add(Conv1D(filters=256, kernel_size=(3), activation='relu', padding='same', name="conv1"))
cnn_mod.add(Dropout(dr))
cnn_mod.add(Conv1D(filters=128, kernel_size=(3), activation='relu', padding='same', name="conv2"))
cnn_mod.add(Dropout(dr))
cnn_mod.add(Flatten())
cnn_mod.add(Dense(256, activation='relu', name="dense"))
cnn_mod.add(Dropout(dr))
cnn_mod.add(Dense(24, activation='softmax', name="output"))
cnn_mod.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_mod.summary()

# Conv2D Convolutional Radio Modulation Recognition Networks
cnn_mod2 = Sequential()
cnn_mod2.add(Reshape((1, 1024, 2), input_shape=(1024, 2)))
cnn_mod2.add(Conv2D(filters=256, kernel_size=(1, 3), strides=1, activation='relu', padding='same', name="conv1"))
cnn_mod2.add(Dropout(dr))
cnn_mod2.add(Conv2D(filters=128, kernel_size=(2, 3), strides=1, activation='relu', padding='same', name="conv2"))
cnn_mod2.add(Dropout(dr))
cnn_mod2.add(Flatten())
cnn_mod2.add(Dense(256, activation='relu', name="dense"))
cnn_mod2.add(Dropout(dr))
cnn_mod2.add(Dense(24, activation='softmax', name="output"))
cnn_mod2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_mod2.summary()

# CNN Over The Air DL
cnn_mod3 = Sequential()
cnn_mod3.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', name="conv1", input_shape=(1024, 2)))
cnn_mod3.add(MaxPooling1D())
cnn_mod3.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', name="conv2"))
cnn_mod3.add(MaxPooling1D())
cnn_mod3.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', name="conv3"))
cnn_mod3.add(MaxPooling1D())
cnn_mod3.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', name="conv4"))
cnn_mod3.add(MaxPooling1D())
cnn_mod3.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', name="conv5"))
cnn_mod3.add(MaxPooling1D())
cnn_mod3.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', name="conv6"))
cnn_mod3.add(MaxPooling1D())
cnn_mod3.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', name="conv7"))
cnn_mod3.add(MaxPooling1D())
cnn_mod3.add(Flatten())
cnn_mod3.add(Dense(128, activation='relu', name="dense1"))
cnn_mod3.add(Dropout(dr))
cnn_mod3.add(Dense(128, activation='relu', name="dense2"))
cnn_mod3.add(Dropout(dr))
cnn_mod3.add(Dense(24, activation='softmax', name="output"))
cnn_mod3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_mod3.summary()

# # training
cnn01_filepath = "cnn01_weights.h5"  # check point filepath
cnn02_filepath = "cnn02_weights.h5"  # check point filepath
cnn03_filepath = "cnn03_weights.h5"  # check point filepath

current_mod = cnn_mod3
current_path = cnn03_filepath

cnn_his = current_mod.fit(X_train, Y_train,
                         batch_size=512,
                         epochs=50,
                         validation_split=0.25,
                         verbose=2,
                         callbacks= [keras.callbacks.ModelCheckpoint(current_path,
                                                                   monitor='val_accuracy',
                                                                   verbose=0,
                                                                   save_best_only=True,
                                                                   mode='auto'),
                                     keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                                   patience=5,
                                                                   verbose=0,
                                                                   mode='auto')
                                     ]
                        )


# re-load weights after training
current_mod.load_weights(current_path)

# predict
# preds = cnn_mod.predict(X_test)
# print(preds)

# evaluation
cnn_score = current_mod.evaluate(X_test, Y_test, verbose=0, batch_size=1)
print(cnn_score)

# plot train vs test accuracy per epoch
plt.figure()
plt.plot(cnn_his.history['loss'])
plt.plot(cnn_his.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'])
plt.show()

plt.figure()
plt.plot(cnn_his.history['accuracy'])
plt.plot(cnn_his.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'])
plt.show()

# Confusion Matrix
test_yhat = current_mod.predict(X_test, batch_size=batch_size)
conf = np.zeros([len(classes), len(classes)])
confnorm = np.zeros([len(classes), len(classes)])
for i in range(0, X_test.shape[0]):
    j = list(Y_test[i, :]).index(1)
    k = int(np.argmax(test_yhat[i, :]))
    conf[j,k] = conf[j,k] + 1
for i in range(0, len(classes)):
    confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])

plt.imshow(confnorm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, labels=(classes), rotation=45)
plt.yticks(tick_marks, labels=(classes))
plt.tight_layout()
plt.ylabel('True')
plt.xlabel('Predicted')

