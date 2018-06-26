# -*- coding:utf-8 -*-
'''
Created on 2018年5月22日

@author: Administrator
'''
from __future__ import print_function
import numpy
import theano.tensor as T
import cv2
import time
import cPickle
numpy.random.seed(1337)  # for reproducibility

from PIL import Image

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D,concatenate 
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K

nb_classes = 2
nb_epoch = 100
batch_size = 10

# input image dimensions
img_rows, img_cols = 143, 143
# number of convolutional filters to use
nb_filters1, nb_filters2, nb_filters3, nb_filters4 = 5, 10, 20, 50
# # size of pooling area for max pooling
nb_pool = 2
# # convolution kernel size
nb_conv = 3

def loadData(dataSet):
    with open(dataSet, 'r') as file:
        train_x, train_y = cPickle.load(file)
        valid_x, valid_y = cPickle.load(file)
        test_x , test_y = cPickle.load(file)
    revl = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    return revl


def Conv2d_BN(x, nb_filter,kernel_size, padding='same',strides=(1,1),name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)  
    x = BatchNormalization(axis=1,name=bn_name)(x)
    '''
    # BN层的作用：
    1、加速收敛:；2、控制过拟合，可以少用或者不用Dropout和正则 ；3、降低网络对初始权重不敏感   4、允许使用较大的学习率
    '''
    return x


def Inception(x,nb_filter):
      
    branch1x1 = Conv2d_BN(x,nb_filter,(1,1), padding='same',strides=1,name=None)  
    
    branch3x3 = Conv2d_BN(x,nb_filter,(1,1), padding='same',strides= 1 ,name=None) 
    branch3x3 = Conv2d_BN(branch3x3,nb_filter,(3,3), padding='same',strides=(1,1),name=None) 
     
    branch5x5 = Conv2d_BN(x,nb_filter,(1,1), padding='same',strides= 1 ,name=None)
    branch5x5 = Conv2d_BN(branch5x5,nb_filter,(5,5), padding='same',strides= 1,name=None)  
  
    branchpool = MaxPooling2D(pool_size=(3,3),strides= 1 ,padding='same')(x)  
    branchpool = Conv2d_BN(branchpool,nb_filter,(1,1),padding='same',strides= 1,name=None)  
  
    x = concatenate([branch1x1,branch3x3,branch5x5,branchpool], axis=3)  
  
    return x

def Net_model():
    
    input = Input(shape=(143,143,1))  
    #padding = 'same'，填充为(步长-1）/2,还可以用ZeroPadding2D((3,3))  
    x = Conv2d_BN(input,16,(3,3),strides=2,padding='same')  
    x = MaxPooling2D(pool_size=(3,3),strides=2,padding='same')(x)  
    x = Conv2d_BN(x,48,(3,3),strides=1,padding='same')  
    x = MaxPooling2D(pool_size=(3,3),strides=2,padding='same')(x)
      
    x = Inception(x,64)#256  
    x = Inception(x,120)#480  
    x = MaxPooling2D(pool_size=(3,3),strides=2,padding='same')(x)  
    x = Inception(x,128)#512  
    x = Inception(x,128)  
    x = Inception(x,128)  
    x = Inception(x,132)#528  
    x = Inception(x,208)#832  
    x = MaxPooling2D(pool_size=(3,3),strides=2,padding='same')(x)  
    x = Inception(x,208)  
    x = Inception(x,256)#1024  
    x = AveragePooling2D(pool_size=(2,2),strides=2,padding='same')(x) 
    
    x = Flatten()(x) 
    x = Dropout(0.5)(x)  
    x = Dense(1000,activation='relu')(x)    
    x = Dense(nb_classes, activation='softmax')(x)
    model = Model(input, x , name='inception')  
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])  
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

if __name__ == '__main__':
    # the data, shuffled and split between tran and test sets
    begin = time.clock()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = loadData('./data/Data2.pkl')
    
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)    
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_val.shape[0], 'validate samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_val = np_utils.to_categorical(y_val, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = Net_model()
    train_model(model, X_train, Y_train, X_val, Y_val)
    
    print('the training is ok!')
    print('the test score is ...')
    
    score = test_model(model, X_test, Y_test)
    
    print('the code is ok')
    end = time.clock()
    print ('the run time is %.2fm :' % ((end - begin)/60.))
