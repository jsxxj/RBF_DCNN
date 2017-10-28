import numpy
weights = numpy.loadtxt('150_weights.txt')
cgm = numpy.loadtxt('150_cgm.txt')
pre_input = numpy.loadtxt('raw_pre_input.txt')

samlength = pre_input.shape[0]
labelNum = weights.shape[0]

classNum = samlength/labelNum

pred_value = numpy.zeros((samlength,labelNum))

for i in range(samlength):
    for j in range(labelNum):
        pred_1 = numpy.dot(pre_input[i,:]-weights[j,:],pre_input[i,:]-weights[j,:])
        pred_2 = pred_1/cgm[j]
        pred_value[i,j] = numpy.exp(-pred_2)

print pred_value

