# coding: utf-8

from LinearRegression import LinearRegression
import numpy as np
import matplotlib.pyplot as plt



numInstance = 100
x = np.random.rand(numInstance, 4)
weight = np.array([3, -2, 16, -3.21]).T
bias = 3.5
y = np.dot(x, weight) + np.random.rand(numInstance) + bias
model = LinearRegression(targetFunc0=np.log, targetFunc1=np.exp)
model.fit(x, y)

y0 = model.predict(x)


print '原始权重：', weight, '  估计权重:', model.weight
print '原始偏置：', bias, '  估计偏置:', model.bias
print '误差均值：', np.mean(y0-np.dot(x, weight)-bias), '偏差均值：', np.mean(np.abs(y0-y))

plt.figure(1)
plt.plot(y, 'ro')
plt.plot(y0, 'b*')
plt.plot(y0-np.dot(x, weight)-bias, 'g.')
plt.legend(('Original', 'Predict', 'Error'))
plt.show()

