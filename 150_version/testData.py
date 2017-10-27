import numpy
c = numpy.loadtxt('150_c.txt')
pre_input = numpy.loadtxt('raw_pre_input.txt')

L_out = numpy.zeros((150,3))
for i in range(50):
    for j in range(3):
        L = numpy.dot(pre_input[i,:] - c[j,:],pre_input[i,:] - c[j,:])
        L_1 = L/c[3,j]
        L_out[i,j] = numpy.exp(-L_1)

for i in range(51,100):
    for j in range(3):
        L = numpy.dot(pre_input[i,:] - c[j,:],pre_input[i,:] - c[j,:])
        L_1 = L/c[3,j]
        L_out[i,j] = numpy.exp(-L_1)

for i in range(101,150):
    for j in range(3):
        L = numpy.dot(pre_input[i,:] - c[j,:],pre_input[i,:] - c[j,:])
        L_1 = L/c[3,j]
        L_out[i,j] = numpy.exp(-L_1)

print L_out
