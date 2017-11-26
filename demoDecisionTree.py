# coding: utf-8

from Common.DecisionTree import DecisionTree
import numpy as np





x = np.random.rand(300, 20)
y = np.floor(np.random.rand(300, 1)*5)  # 类别 0 1 2 3 4
model = DecisionTree()
model.fit(x, y)
print 'Model Success!'
print '预测', model.predict(x[4:8, :])
print '真实', y[4:8]
a = 0








x = np.random.rand(200, 7)
x[:, 3] = np.floor(x[:, 3]*5)
y = np.floor(np.random.rand(200, 1)*5)  # 类别 0 1 2 3 4
attrIsCont = [1, 1, 1, 0, 1, 1, 1]
model = DecisionTree()
model.fit(x, y, attrIsCont=attrIsCont)
print 'Model Success!'
print '预测', model.predict(x[4:8, :])
print '真实', y[4:8]
a = 0

