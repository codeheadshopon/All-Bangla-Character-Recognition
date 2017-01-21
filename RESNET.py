from __future__ import print_function
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from PIL import Image
from os import listdir
from keras.models import load_model
from os.path import isfile, join
import PIL.ImageOps
import matplotlib.cm as cm
import numpy as np
from skimage import color
from skimage import io
import pickle
#import cv
import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras.regularizers import l2, activity_l2
from keras.callbacks import TensorBoard
import cv2
import numpy as np
import gzip,cPickle,sys
from resnet import Residual
from skimage.transform import resize
import scipy

batch_size = 128
nb_classes = 231
nb_epoch = 100

img_rows, img_cols = 28, 28
pool_size = (2, 2)
kernel_size = (3, 3)

# Augmentation Fllag  #
isAugment=False


def dataset_load(path):
    if path.endswith(".gz"):
        f=gzip.open(path,'rb')
    else:
        f=open(path,'rb')

    if sys.version_info<(3,):
        data=cPickle.load(f)
    else:
        data=cPickle.load(f,encoding="bytes")
    f.close()
    return data

(X_train,y_train),(X_test,y_test)=dataset_load('./FULL_BANGLA.pkl.gz')

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

if isAugment:
    test_images = []
    test_labels = []

    RESH=25
    for i in range(X_train.shape[0]):
            test_images.append(X_train[i])
            test_labels.append(y_train[i])

            img = X_train[i].reshape(28, 28)
            resized = resize(img, (RESH, RESH))
            resized_updated2 = resize(resized, (28, 28))

            test_images.append(resized_updated2.reshape(1, 28, 28))
            test_labels.append(y_train[i])



    X_train = np.asarray(test_images)
    print(X_train.shape)
    y_train = np.asarray(test_labels)
    print(y_train.shape)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

input_shape = (1, img_rows, img_cols)

input_var = Input(shape=input_shape)

CONV_1 = Convolution2D(64, kernel_size[0], kernel_size[1],
                      border_mode='same', activation='relu')(input_var)

CONV_2 = Convolution2D(16, kernel_size[0], kernel_size[1],
                      border_mode='same', activation='relu')(CONV_1)

resnet = CONV_2

for _ in range(6):
    resnet = Residual(Convolution2D(16, kernel_size[0], kernel_size[1],
                                  border_mode='same'))(resnet)
    resnet = Residual(Convolution2D(16, kernel_size[0], kernel_size[1],
                                    border_mode='same'))(resnet)
    resnet = Residual(Convolution2D(16, kernel_size[0], kernel_size[1],
                                    border_mode='same'))(resnet)

    resnet = Activation('relu')(resnet)

mxpool = MaxPooling2D(pool_size=pool_size)(resnet)
flat = Flatten()(mxpool)
dropout = Dropout(0.5)(flat)
softmax = Dense(nb_classes, activation='softmax')(dropout)

model = Model(input=[input_var], output=[softmax])
model.compile(loss='categorical_crossentropy',
              optimizer='RMSprop',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
model.save('mnist_model.h5')
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])