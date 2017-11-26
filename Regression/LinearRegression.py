# coding: utf-8
# 广义线性模型
# 一行一个样本

from Common.BasicModel import BasicModel
import numpy as np


class LinearRegression(BasicModel):
    def __init__(self, targetFunc0 = None, targetFunc1 = None):
        self.targetFunc0 = targetFunc0  # 广义线性模型的函数，将施加于 target，必须能处理numpy向量
        self.targetFunc1 = targetFunc1  # 互为反函数

    def fit(self, x, y):
        # 采用targetFunc处理预测目标
        if self.targetFunc0 is None:
           tY = self.DoTargetTransform(y, 0)
        else:
           tY = y
        numVar = x.shape[1]+1  # 变量的数目
        tX = np.hstack((x, np.ones([len(y), 1])))
        weight = np.dot(np.dot(np.linalg.inv(np.dot(tX.T, tX)), tX.T), tY)
        self.weight = weight[0:numVar-1]
        self.bias = weight[numVar-1]
        self.__weight = weight

    def predict(self, x):
        tY = np.dot(x, self.weight) + self.bias
        if self.targetFunc1 is None:
           return self.DoTargetTransform(tY, 1)
        else:
           return tY

    def DoTargetTransform(self, y, idxFunc):
        numInstances = len(y)
        tY = np.zeros(y.shape)
        if idxFunc == 0:
            for i in range(numInstances):
                tY[i] = self.targetFunc0(y[i])
        elif idxFunc == 1:
            for i in range(numInstances):
                tY[i] = self.targetFunc1(y[i])
        else:
            raise ValueError('Wrong idxFunc')
        return tY


