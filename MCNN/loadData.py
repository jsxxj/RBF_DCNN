# -*- coding:utf-8 -*-
'''
Created on 2018年6月3日

@author: Administrator
'''
import numpy
import os

def loadData(path):
    rng = numpy.random.randint(23455)
    files = os.listdir(path)
    for file in files:
        fname = os.path.splitext(file)
        fr = os.path.join(path, file)
        if fname[0].split('_')[1] == 'TRAIN':
            with open(fr, 'r') as f:
                lines = f.readlines()
                data = numpy.zeros((len(lines),len(lines[0].split(','))))
                i = 0
                for line in lines:
                    line = line.strip()
                    line = line.split(',')
                    data[i, 1:] = line[1:]
                    data[i, 0] = int(line[0]) - 1
                    i += 1
            train_data = os.getcwd() + '/txt/' + fname[0].split('_')[0] + '_train.txt'
            train_label = os.getcwd() + '/txt/' + fname[0].split('_')[0] + '_train_label.txt'
            Tdata = os.getcwd() + '/txt/' + fname[0].split('_')[0] + '_Train.txt'
            
            data = data[data[:,0].argsort()]
            train_x = data[:, 1:].astype(numpy.float32)
            print train_x.shape
            train_y = numpy.int_(data[:, 0].astype(numpy.float32)) - 1
            numpy.savetxt(Tdata, data, fmt='%s', newline = '\n')
#             numpy.savetxt(train_data, train_x, fmt='%s', newline = '\n')
#             numpy.savetxt(train_label, train_y, fmt='%s', newline = '\n')
            
        else:
            with open(fr, 'r') as f:
                lines = f.readlines()
                data = numpy.zeros((len(lines),len(lines[0].split(','))))
                i = 0
                for line in lines:
                    line = line.strip()
                    line = line.split(',')
                    data[i, 1:] = line[1:]
                    data[i, 0] = int(line[0]) - 1
                    i += 1
                test_data = os.getcwd() + '/txt/' + fname[0].split('_')[0] + '_test.txt'
                test_label = os.getcwd() + '/txt/' + fname[0].split('_')[0] +'_test_label.txt'
                Tdata = os.getcwd() + '/txt/' + fname[0].split('_')[0] + '_Test.txt'
                data = data[data[:,0].argsort()]
                test_x = data[:, 1:].astype(numpy.float32)
                test_y = numpy.int_(data[:, 0].astype(numpy.float32)) - 1
                numpy.savetxt(Tdata, data, fmt='%s', newline = '\n')
#                 numpy.savetxt(test_data, test_x, fmt='%s', newline = '\n')
#                 numpy.savetxt(test_label, test_y, fmt='%s', newline ='\n')     
                      
#     m,n = train_x.shape
#     ind = numpy.arange(m)
#     rng.shuffle(ind)

path = './data/' 
loadData(path)
print 'the data is ok!!!'


    
