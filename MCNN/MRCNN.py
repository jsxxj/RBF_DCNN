# -*- coding:utf-8 -*-
from __future__ import print_function
import numpy
import cPickle

numpy.random.seed(1337)
from PIL import Image

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, concatenate, add, Multiply, BatchNormalization , AveragePooling2D, ZeroPadding2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K

nb_classes = 2
nb_epoch = 100
batch_size = 10

img_rows, img_cols = 143, 143
nb_filter1, nb_filter2, nb_filter3, nb_filter4 = 8, 16, 32, 64
nb_pool = 2
nb_conv = 3

def loadData(dataSet):
    with open(dataSet, 'r') as file:
        train_x, train_y = cPickle.load(file)
        valid_x, valid_y = cPickle.load(file)
        test_x, test_y = cPickle.load(file)
    
    rval = [(train_x, train_y), (valid_x, valid_y),
            (test_x, test_y)]
    
    return rval

def splitData(data):
    samples, size = data.shape
    channel = 3    
    matrix1 = data[:,0*(size/channel):1*size/channel]
    matrix2 = data[:,1*(size/channel):2*size/channel]
    matrix3 = data[:,2*(size/channel):3*size/channel]
    return matrix1, matrix2, matrix3

def Conv2d_BN(x, nb_filter,kernel_size, strides=(1,1), padding='same',name=None):  
    if name is not None:  
        bn_name = name + '_bn'  
        conv_name = name + '_conv'  
    else:  
        bn_name = None  
        conv_name = None  
  
    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)  
    x = BatchNormalization(axis= 3, name=bn_name)(x) 
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
  
def Conv_Block(inpt,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut=False):  
    x = Conv2d_BN(inpt,nb_filter=nb_filter,kernel_size=kernel_size,strides=strides,padding='same')  
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size,padding='same')  
    if with_conv_shortcut:  
        shortcut = Conv2d_BN(inpt,nb_filter=nb_filter,strides=strides,kernel_size=kernel_size)  
        x = add([x,shortcut])  
        return x  
    else:  
        x = add([x,inpt])  
        return x  
        
def ConvPooling(input): 
    #建立padding画布，主要适配不同图片尺寸或者避免边缘特征磨损的情况
    x = ZeroPadding2D((3,3))(input)  
    x = Conv2d_BN(x,nb_filter=8,kernel_size=(3,3),strides=(2,2),padding='valid')  
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(x)  
    #(56,56,64)  
    x = Conv_Block(x,nb_filter=8,kernel_size=(3,3))  
    x = Conv_Block(x,nb_filter=8,kernel_size=(3,3))
    #x = Inception(x, 8) 
    #x = Conv_Block(x,nb_filter=8,kernel_size=(3,3))  
    #(28,28,128)  
    x = Conv_Block(x,nb_filter=16,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)  
    x = Conv_Block(x,nb_filter=16,kernel_size=(3,3))
    #x = Inception(x, 16)  
    #x = Conv_Block(x,nb_filter=16,kernel_size=(3,3))  
    #x = Conv_Block(x,nb_filter=16,kernel_size=(3,3))  
    #(14,14,256)  
    
    x = Conv_Block(x,nb_filter=32,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)  
    x = Conv_Block(x,nb_filter=32,kernel_size=(3,3)) 
    #x = Inception(x, 32)
    #x = AveragePooling2D(pool_size=(2,2))(x)
    
    #x = Conv_Block(x,nb_filter=32,kernel_size=(3,3))  
    #x = Conv_Block(x,nb_filter=32,kernel_size=(3,3))  
    #x = Conv_Block(x,nb_filter=32,kernel_size=(3,3))  
    #x = Conv_Block(x,nb_filter=32,kernel_size=(3,3))  
    #(7,7,512)  
    x = Conv_Block(x,nb_filter=64,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)  
    x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))
    #x = Inception(x, 64)  
    #x = Conv_Block(x,nb_filter=64,kernel_size=(3,3)) 
    
    x = Conv2D(64,kernel_size=(3,3),padding='same',strides=(1,1),activation='relu')(x)
    x = AveragePooling2D(pool_size=(2,2))(x) 
    
    return x 
    

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
    #x = Conv2d_BN(merged_input,64,nb_conv, strides=(1,1), padding='same')
    
    x = Dense(1000, activation='relu', name = "Dense_1")(merged_input)
    x = Dropout(0.5)(x)
    
    predictions = Dense(nb_classes, activation='softmax', name = "output_layer")(x)

    model = Model(inputs=[inputs_1, inputs_2, inputs_3], outputs= predictions)
    
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
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
    
