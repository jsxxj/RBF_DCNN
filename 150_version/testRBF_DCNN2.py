import numpy

classNum = 3

pre_input = numpy.loadtxt('150_pre_input.txt')
pro_value = numpy.loadtxt('150_pro_value.txt')

samlenth,dimi = pre_input.shape

'''calc the variance'''
x0 = numpy.zeros((50,dimi))
for i in range(50):
    x0[i,:] = pre_input[i,:]
w_cov = numpy.zeros(dimi)
for j in range(dimi):
    w_cov[j] = numpy.cov(x0[:,j])

alpha1 = numpy.dot(w_cov,w_cov)

x0 = numpy.zeros((50,dimi))
for i in range(51,100):
    x0[i-51,:] = pre_input[i,:]
w_cov = numpy.zeros(dimi)
for j in range(dimi):
    w_cov[j] = numpy.cov(x0[:,j])
alpha2 = numpy.dot(w_cov,w_cov)

x0 = numpy.zeros((50,dimi))
for i in range(101,150):
    x0[i-101,:] = pre_input[i,:]
w_cov = numpy.zeros(dimi)
for j in range(dimi):
    w_cov[j] = numpy.cov(x0[:,j])
alpha3 = numpy.dot(w_cov,w_cov)


'''get the maxpro'''
#max_pro = numpy.max(pro_value,axis=1)

'''get the weights input'''
c0 = numpy.zeros(dimi)
for i in range(50):
    c0[:] += pre_input[i,:]*max(pro_value[i])/50

c1 = numpy.zeros(dimi)
for i in range(51,100):
    c1[:] += pre_input[i,:]*max(pro_value[i])/50

c2 = numpy.zeros(dimi)
for i in range(101,150):
    c2[:] += pre_input[i,:]*max(pro_value[i])/50

c3 = numpy.zeros(dimi)
c3[0] += alpha1*50
c3[1] += alpha2*50
c3[2] += alpha3*50

c = [c0,c1,c2,c3]

numpy.savetxt('150_c.txt',c,fmt='%s',newline='\n')











