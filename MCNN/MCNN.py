# -*- coding:utf-8 -*-
from __future__ import print_function
import numpy
import theano.tensor as T
import cv2
import cPickle

numpy.random.seed(1337)  # for reproducibility

from PIL import Image

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, concatenate, add, Multiply
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K

        
nb_classes = 2
nb_epoch = 100
batch_size = 10

# input image dimensions
img_rows, img_cols = 143, 143
# number of convolutional filters to use
nb_filters1, nb_filters2, nb_filters3, nb_filters4 = 8, 16, 32, 64
# # size of pooling area for max pooling
nb_pool = 2
# # convolution kernel size
nb_conv = 3

def splitData(data):
    samples, size = data.shape
    channel = 3    
    matrix1 = data[:,0*(size/channel):1*size/channel]
    matrix2 = data[:,1*(size/channel):2*size/channel]
    matrix3 = data[:,2*(size/channel):3*size/channel]
    return matrix1, matrix2, matrix3
    
def loadData(dataSet):
    with open(dataSet, 'r') as file:
        train_x, train_y = cPickle.load(file)
        valid_x, valid_y = cPickle.load(file)
        test_x, test_y = cPickle.load(file)
    
    rval = [(train_x, train_y), (valid_x, valid_y),
            (test_x, test_y)]
    
    return rval

def ConvPooling(input):

    Conv = Conv2D(nb_filters1,(nb_conv, nb_conv), padding='same')
    conv1 = Conv(input)
    Act = Activation('relu')
    act1 = Act(conv1)
    maxpool = MaxPooling2D((nb_pool, nb_pool))
    pool1 = maxpool(act1)

    Conv2 = Conv2D(nb_filters2, (nb_conv, nb_conv), padding='same')
    conv2 = Conv2(pool1)
    Act2 = Activation('relu')
    act2 = Act2(conv2)
    maxpool2 = MaxPooling2D((nb_pool, nb_pool))
    pool_2 = maxpool2(act2)
    
    Conv3 = Conv2D(nb_filters3, (nb_conv, nb_conv), padding='same')
    conv3 = Conv3(pool_2)
    Act3 = Activation('relu')
    act3 = Act3(conv3)
    maxpool3 = MaxPooling2D((nb_pool, nb_pool))
    pool_3 = maxpool3(act3)
    
    Conv4 = Conv2D(nb_filters4, (nb_conv, nb_conv), padding='same')
    conv4 = Conv4(pool_3)
    Act4 = Activation('relu')
    act4 = Act4(conv4)
    maxpool4 = MaxPooling2D((nb_pool, nb_pool))
    pool_4 = maxpool4(act4)
    
    return pool_4


def Net_model(lr=0.005, decay=1e-6, momentum=0.9):
    
    inputs_1 = Input(shape=(img_rows, img_cols, 1), name = 'input_1')
    inputs_2 = Input(shape=(img_rows, img_cols, 1), name = 'input_2')
    inputs_3 = Input(shape=(img_rows, img_cols, 1), name = 'input_3')
    
    pool_2_1 = ConvPooling(inputs_1)
    pool_2_2 = ConvPooling(inputs_2)
    pool_2_3 = ConvPooling(inputs_3)
    
    input_1 = Flatten( name='flatten_1' )(pool_2_1)
    input_2 = Flatten( name='flatten_2' )(pool_2_2)
    input_3 = Flatten( name='flatten_3' )(pool_2_3)
    
    merged_input = add([input_1, input_2, input_3], name = "merged_layer")
    
    x = Dense(1000, activation='relu', name = "Dense_1")(merged_input)
    x = Dropout(0.5)(x)
    
    predictions = Dense(nb_classes, activation='softmax', name = "output_layer")(x)

    model = Model(inputs=[inputs_1, inputs_2, inputs_3], outputs= predictions)
    
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # optimizer =: rmsprop or  sgd 
    # 优化器  http://keras-cn.readthedocs.io/en/latest/other/optimizers/
    
    return model
    
    
def train_model(model, X_train, Y_train, X_val, Y_val):
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(X_val, Y_val))
    model.save_weights('./model_weight/model_weights.h5', overwrite=True)


def test_model(model, X, Y):
    model.load_weights('./model_weight/model_weights.h5')
    score = model.evaluate(X, Y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return score
    
def predict_model(model, X):
    predictions = model.predict(X)
    pred = theano.tensor.argmax(predictions,axis = 1)
    return pred



if __name__ == '__main__':
    # the data, shuffled and split between tran and test sets
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = loadData('./data/Data2.pkl')
    
    X_train1, X_train2, X_train3 = splitData(X_train)
    X_val1, X_val2, X_val3  = splitData(X_val)
    X_test1, X_test2, X_test3 = splitData(X_test)
    
    X_train1 = X_train1.reshape(X_train1.shape[0], img_rows, img_cols, 1)
    X_train2 = X_train2.reshape(X_train2.shape[0], img_rows, img_cols, 1)
    X_train3 = X_train3.reshape(X_train3.shape[0], img_rows, img_cols, 1)
    
    X_val1 = X_val1.reshape(X_val1.shape[0], img_rows, img_cols, 1)
    X_val2 = X_val2.reshape(X_val2.shape[0], img_rows, img_cols, 1)
    X_val3 = X_val3.reshape(X_val3.shape[0], img_rows, img_cols, 1)
    
    X_test1 = X_test1.reshape(X_test1.shape[0], img_rows, img_cols, 1)
    X_test2 = X_test2.reshape(X_test2.shape[0], img_rows, img_cols, 1)
    X_test3 = X_test3.reshape(X_test3.shape[0], img_rows, img_cols, 1)
    
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_val.shape[0], 'validate samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_val = np_utils.to_categorical(y_val, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    
    X_train = [X_train1, X_train2, X_train3]
    X_val = [X_val1, X_val2, X_val3]

    model = Net_model()
    train_model(model, X_train, Y_train, X_val, Y_val)
    
    print('the training is ok!')
    print('the test score is ...')
    
    X_test = [X_test1, X_test2, X_test3]
    
    score = test_model(model, X_test, Y_test)
    #prob = model.predict(X_train)
    #print (prob)
    
    print('the code is ok')
    
