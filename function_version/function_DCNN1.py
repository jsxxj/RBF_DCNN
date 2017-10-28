#-*-coding:utf-8-*-
import numpy
def calcWeights(pre_input,pro_value):

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

    #numpy.savetxt('150_cgm.txt',cgm,fmt='%s',newline='\n')

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

    #numpy.savetxt('150_weights.txt',final_weights,fmt='%s',newline='\n')
    return cgm,final_weights

def pred_value(pre_input,cgm,weights):

    samlength = pre_input.shape[0]
    labelNum = weights.shape[0]

    classNum = samlength / labelNum

    pred_value = numpy.zeros((samlength, labelNum))

    for i in range(samlength):
        for j in range(labelNum):
            pred_1 = numpy.dot(pre_input[i, :] - weights[j, :], pre_input[i, :] - weights[j, :])
            pred_2 = pred_1 / cgm[j]
            pred_value[i, j] = numpy.exp(-pred_2)

    #print pred_value
    return pred_value

if __name__ == '__main__':
     pre_input = numpy.loadtxt('150_pre_input.txt')
     pro_value = numpy.loadtxt('150_pro_value.txt')

     cgm,weights = calcWeights(pre_input, pro_value)
     pred_value = pred_value(pre_input,cgm,weights)

     print pred_value



