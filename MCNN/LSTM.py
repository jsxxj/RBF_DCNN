# -*- coding:utf-8 -*-
from __future__ import print_function
import numpy
import theano.tensor as T
import cv2
import cPickle
numpy.random.seed(1337)  # for reproducibility

from PIL import Image

from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, LSTM 
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K

img_rows, img_cols = 143, 143
nb_classes = 2
nb_epoch = 60
batch_size = 10

def loadData(file):
    with open(file, 'r') as f:
        train_x, train_y = cPickle.load(f)
        valid_x, valid_y = cPickle.load(f)
        test_x, test_y = cPickle.load(f)
    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    return rval

'''
def Net_model():
    
    input = Input(shape=(img_rows, img_cols))
    
    lstm = LSTM(64)
    #lstm = LSTM(32, return_state=True)(lstm)
    #lstm = LSTM(32, return_state=True)(lstm)
    
    hidden = Dense(1000, activation='relu')
    out = Dropout(0.5)
    predictions = Dense(nb_classes, activation = 'softmax')
    
    model = Model(inputs = input, output=predictions)
    model.compile(optimizer=['rmsprop'],
                  loss = 'categorical_crossentropy',
                  metrix = ['accuracy'])
    return model 
'''
    
def Net_model():
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=(img_rows, img_cols)))
    model.add(LSTM(128,return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(256))
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax', name='output_layer'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer = 'rmsprop',
                  metrics = ['accuracy']
                  )
    return model
     

def train_model(model, x_train, y_train, x_val, y_val):
    model.fit(x_train, y_train, batch_size = batch_size, epochs = nb_epoch,
              verbose=1, validation_data=(x_val, y_val))
    model.save_weights('./weights/model_weights.h5', overwrite = True)
    
    
def test_model(model, x_test, y_test):
    model.load_weights('./weights/model_weights.h5')
    score = model.evaluate(model, x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return score
    
    
if __name__ == '__main__':
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = loadData('./data/Data2.pkl')
    
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols)
    X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols)
    
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_val.shape[0], 'validate samples')
    print(X_test.shape[0], 'test samples')
    
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_val = np_utils.to_categorical(Y_val, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    
    model = Net_model()
    train_model(model, X_train, Y_train, X_val, Y_val)
    print (' the code is ok!!!')
