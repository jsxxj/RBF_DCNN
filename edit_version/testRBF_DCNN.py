# -*- coding: utf-8 -*-
import theano
import sys
import math
from theano.tensor.nnet import conv
import theano.tensor as T
import numpy
import time,cPickle
from theano.tensor.signal import downsample

def load_params(filename):
    f = open(filename,'rb')
    layer0_params = cPickle.load(f)
    layer1_params = cPickle.load(f)
    layer2_params = cPickle.load(f)
    layer3_params = cPickle.load(f)
    return layer0_params, layer1_params, layer2_params, layer3_params

def load_data(dataset):

    print('... loading data')

    f = open(dataset, 'rb')
    test_set = cPickle.load(f)

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)

    rval = [(test_set_x, test_set_y)]
    return rval

class LeNetConvPoolLayer(object):
    def __init__(self, input, params_W, params_b, filter_shape, image_shape, poolsize=(2, 2)):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.W = params_W
        self.b = params_b
        # 卷积
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )
        # 子采样
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]


class HiddenLayer(object):
    def __init__(self, input, params_W, params_b, n_in, n_out,
                 activation=T.tanh):
        self.input = input
        self.W = params_W
        self.b = params_b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]


class LogisticRegression(object):
    def __init__(self, input, params_W, params_b, n_in, n_out):
        self.input = input
        self.W = params_W
        self.b = params_b
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class RBF(object):
    def __init__(self,pre_input,pro_value):
        self.pre_input = pre_input
        self.pro_value = pro_value


def test_CNN(dataset='testData5.pkl', params_file='Dropout_2_2_CNN_params.pkl'):
    dataset = 'testData5.pkl'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[0]
    test_set_x = test_set_x.get_value()
    sampleLength = len(test_set_x)
    print sampleLength
    layer0_params, layer1_params, layer2_params, layer3_params = load_params(params_file)

    x = T.matrix()

    label_y = []
    for i in range(len(test_set_x)):
        if i < 50:
            label_y.append(0)
        elif i >= 50 and i < 100:
            label_y.append(1)
        else:
            label_y.append(2)

    print '... testing the model ...'

    # transfrom x from (batchsize, 28*28) to (batchsize,feature,28,28))
    # I_shape = (28,28),F_shape = (5,5),
    # 第一层卷积、池化后  第一层卷积核为20个，每一个样本图片都产生20个特征图，
    N_filters_0 = 20
    D_features_0 = 1
    # 输入必须是为四维的，所以需要用到reshape，这一层的输入是一批样本是20个样本，28*28，

    layer0_input = x.reshape((sampleLength, 1, 40, 36))

    layer0 = LeNetConvPoolLayer(
        input=layer0_input,
        params_W=layer0_params[0],
        params_b=layer0_params[1],
        image_shape=(sampleLength, 1, 40, 36),
        filter_shape=(N_filters_0, 1, 5, 5),
        poolsize=(2, 2)
    )
    # layer0.output: (batch_size, N_filters_0, (40-5+1)/2, (36-5+1)/2) -> 20*20*18*16

    N_filters_1 = 50
    D_features_1 = N_filters_0
    layer1 = LeNetConvPoolLayer(
        input=layer0.output,
        params_W=layer1_params[0],
        params_b=layer1_params[1],
        image_shape=(sampleLength, N_filters_0, 18, 16),
        filter_shape=(N_filters_1, D_features_1, 5, 5),
        poolsize=(2, 2)
    )
    # layer1.output: (20,50,7,6)
    # 第二层输出为20个样本，每一个样本图片对应着50张4*4的特征图，其中的卷积和池化操作都是同第一层layer0是一样的。
    # 这一层是将上一层的输出的样本的特征图进行一个平面化，也就是拉成一个一维向量，最后变成一个20*800的矩阵，每一行代表一个样本，

    # (20,50,4,4)->(20,(50*4*4))
    layer2_input = layer1.output.flatten(2)
    # 上一层的输出变成了20*800的矩阵，通过全连接，隐层操作，将800变成了500个神经元，里面涉及到全连接。
    layer2 = HiddenLayer(
        layer2_input,
        params_W=layer2_params[0],
        params_b=layer2_params[1],
        n_in=50 * 7 * 6,
        n_out=500,
        activation=T.tanh
    )

    # 这里为逻辑回归层，主要是softmax函数作为输出，
    layer3 = LogisticRegression(input=layer2.output,
                                params_W=layer3_params[0],
                                params_b=layer3_params[1],
                                n_in=500,
                                n_out=3)
    '''预测函数'''
    f_pred = theano.function(
        [x],
        layer3.p_y_given_x,
        allow_input_downcast=True
    )
    '''the output of the second convolutionPooling layer'''
    f_layer2_input = theano.function(
        [x],
        layer2.input,
        allow_input_downcast=True
    )

    '''get the softmax pro-value'''
    pro_value = f_pred(test_set_x[:])
    pre_input = f_layer2_input(test_set_x[:])

    print pre_input.shape
    
    numpy.savetxt('150_pro_value.txt',pro_value,fmt='%s',newline='\n')
    numpy.savetxt('150_pre_input.txt',pre_input,fmt='%s',newline='\n')

    print '...the input and provalue is saved...'


if __name__ == '__main__':
    test_CNN()
