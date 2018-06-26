#coding=utf-8  
from keras.models import Model  
from keras.layers import Input,Dense,Dropout,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D,concatenate,Activation,ZeroPadding2D  
from keras.layers import add,Flatten 
from keras.optimizers import SGD
from keras.utils import np_utils 
#from keras.layers.convolutional import Conv2D,MaxPooling2D,AveragePooling2D  
import numpy as np  
import cPickle

seed = 7  # 23455  
np.random.seed(seed)  

img_rows, img_cols = 143, 143
nb_epochs = 50
batch_size = 10
nb_classes = 2

def loadData(file):
    with open(file, 'r') as f:
        train_x, train_y = cPickle.load(f)
        valid_x, valid_y = cPickle.load(f)
        test_x, test_y = cPickle.load(f)
    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    return rval
  
def Conv2d_BN(x, nb_filter,kernel_size, strides=(1,1), padding='same',name=None):  
    if name is not None:  
        bn_name = name + '_bn'  
        conv_name = name + '_conv'  
    else:  
        bn_name = None  
        conv_name = None  
  
    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)  
    x = BatchNormalization(axis=3,name=bn_name)(x)  
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

def Net_model():
    
    input = Input(shape=(img_rows,img_cols,1), name = 'input_layer')  
    x = ZeroPadding2D((3,3))(input)  
    x = Conv2d_BN(x,nb_filter=8,kernel_size=(3,3),strides=(2,2),padding='valid')  
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)  
    #(56,56,64)  
    x = Conv_Block(x,nb_filter=8,kernel_size=(3,3))  
    x = Conv_Block(x,nb_filter=8,kernel_size=(3,3))  
    #x = Conv_Block(x,nb_filter=8,kernel_size=(3,3))  
    #(28,28,128)  
    x = Conv_Block(x,nb_filter=16,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)  
    x = Conv_Block(x,nb_filter=16,kernel_size=(3,3))  
    #x = Conv_Block(x,nb_filter=16,kernel_size=(3,3))  
    #x = Conv_Block(x,nb_filter=16,kernel_size=(3,3))  
    #(14,14,256)  
    x = Conv_Block(x,nb_filter=32,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)  
    x = Conv_Block(x,nb_filter=32,kernel_size=(3,3))  
    #x = Conv_Block(x,nb_filter=32,kernel_size=(3,3))  
    #x = Conv_Block(x,nb_filter=32,kernel_size=(3,3))  
    #x = Conv_Block(x,nb_filter=32,kernel_size=(3,3))  
    #x = Conv_Block(x,nb_filter=32,kernel_size=(3,3))  
    #(7,7,512)  
    x = Conv_Block(x,nb_filter=64,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)  
    x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))  
    #x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))
    x = Conv2D(64,kernel_size=(3,3),padding='same',strides=(1,1),activation='relu')(x)  
    x = AveragePooling2D(pool_size=(2,2))(x)  
    x = Flatten()(x)  
    x = Dense(1000,activation='relu', name = 'feature_layer')(x)  
    x = Dense(nb_classes,activation='softmax')(x) 
    model = Model(inputs=input,outputs=x)  
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])  
    return model

def train_model(model, x_train, y_train, x_valid, y_valid):
    model.fit(x_train, y_train, batch_size = batch_size, epochs = nb_epochs,
              verbose = 1, validation_data=(x_valid, y_valid)) 
    model.save_weights('./weights/model_weights.h5', overwrite = True)
    
def test_model(model, x_test, y_test):
    model.load_weights('./weights/model_weights.h5')
    score = model.evaluate(model, x_test, y_test)
    print ('Test loss:', score[0])
    print ('Test accuracy', score[1])
    
if __name__ == '__main__':

    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = loadData('./data/Data2.pkl')
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    
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
