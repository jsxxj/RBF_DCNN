# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:49:12 2017

@author: jsx
"""
import numpy
import os

def process():
    filePath = os.getcwd()
    files = os.listdir(filePath)
    for file in files:
        fr = os.path.join(filePath,file)
        f = numpy.loadtxt(fr)
        m, n = f.shape
        data = numpy.zeros((m,n))
        differ = numpy.zeros(m)
        for i in range(m):
            '''
            (f[i]).sort()从大到小
            sorted(f[i])：从小到大
            
            '''
            data[i,:] = sorted(f[i,:])
            #print data[i,-1]
            differ[i] = data[i,-1] - data[i,-2]
        newName = 'new'+ '_' + file
        numpy.savetxt(newName,data,fmt='%s',newline='\n')
        numpy.savetxt('differ.txt',differ,fmt='%s',newline='\n')
        
            
        '''
        with open(fr,'r') as f:
            lines =f.readlines()
        for line in lines:
            print sorted(line)
        '''
    

if __name__ == '__main__':
    process()
    print '...done...'
