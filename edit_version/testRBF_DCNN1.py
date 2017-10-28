#-*-coding:utf-8-*-
import numpy

pre_input = numpy.loadtxt('150_pre_input.txt')
pro_value = numpy.loadtxt('150_pro_value.txt')

samLength,Dimi = pre_input.shape
labelNum = pro_value.shape[1]
classNum = samLength/labelNum

'''clac the cigema'''


w = numpy.zeros((labelNum,Dimi))
for i in range(labelNum):
    for j in range(Dimi):
        w[i,j] = numpy.cov(pre_input[i*classNum:(i+1)*classNum,j])

cgm = numpy.zeros(labelNum)

for i in range(labelNum):
    cgm[i] = numpy.dot(w[i,:],w[i,:])*classNum

print cgm

numpy.savetxt('150_cgm.txt',cgm,fmt='%s',newline='\n')


'''clac the weights'''
weights = numpy.zeros((samLength,Dimi))
final_weights = numpy.zeros((labelNum,Dimi))
for i in range(samLength):
    weights[i,:] = pre_input[i,:]*max(pro_value[i,:])

'''
axis=0表述列
axis=1表述行
'''

for i in range(labelNum):
    final_weights[i,:] = numpy.mean(weights[i*classNum:(i+1)*classNum,:],axis=0)

numpy.savetxt('150_weights.txt',final_weights,fmt='%s',newline='\n')







