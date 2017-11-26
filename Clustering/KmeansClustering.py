# coding: utf-8

import numpy as np
import numpy.linalg as lin

from Common.BasicModel import BasicModel


class KmeansClustering(BasicModel):
    def __init__(self, numCluster=3, distanceType='euclidean', distancePara=[], maxIter=100, stopError = 1e-3):
        self.numCluster = numCluster
        self.distanceType = distanceType
        self.distancePara = distancePara
        self.maxIter = maxIter
        self.stopError = stopError

    def fit(self, x):
        numInstance = x.shape[0]
        # 初始化均值向量
        meanVec = x[np.random.permutation(numInstance)[0:self.numCluster], :]
        isContinue = True
        disToMeanVec = np.zeros([numInstance, self.numCluster])  # 一行为一个样本到各均值的距离
        meanVecOld = meanVec  # 前一次均值向量
        iterCount = 0
        while isContinue:
            # 计算到均值向量的距离
            for i in range(numInstance):
                for j in range(self.numCluster):
                    if self.distanceType == 'euclidean':
                        disToMeanVec[i, j] = lin.norm(x[i, :] - meanVec[j, :])
                    else:
                        raise Exception
            # 为每个样本划分类别
            clusterIdx = np.argmin(disToMeanVec, axis=1)  # 每行距离最小的位置
            # 更新均值向量
            for i in range(self.numCluster):
                meanVec[i, :] = np.mean(x[clusterIdx == i, :], axis=0)
            # 判断是否结束
            iterCount += 1
            if self.maxIter < iterCount or np.sum(np.abs(meanVec-meanVecOld)) < self.stopError:
                isContinue = False
        return clusterIdx

