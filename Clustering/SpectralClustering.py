# coding: utf-8

import numpy as np
import numpy.linalg as lin

from KmeansClustering import KmeansClustering
from Common.BasicModel import BasicModel


class SpectralClustering(BasicModel):
    def __init__(self, graphWay='full', graphValue=0.4, numCluster=3):
        self.graphWay = graphWay
        self.graphValue = graphValue
        self.numCluster = numCluster

    def fit(self, x):
        self.x = x
        self.makeSimilarity()  # 构造相似矩阵
        print self.graphWay
        self.makeGraph()  # 构造近邻矩阵
        print "计算拉普拉斯矩阵"
        self.calLaplaceMat()  # 计算拉普拉斯矩阵
        print "计算最小特征值对应的k个特征向量"
        self.calEigenMat()  # 计算最小特征值对应的k个特征向量
        print "对特征向量矩阵进行Kmean聚类"
        self.doKmeans()  # 对特征向量矩阵进行Kmean聚类


    def makeSimilarity(self):  # 采用高斯距离计算相似度
        numInstances = self.x.shape[0]
        self.simMat = np.zeros([numInstances, numInstances])
        for i in range(numInstances - 1):
            for j in range(i + 1, numInstances):
                self.simMat[i, j] = np.exp(
                    -1 * lin.norm(self.x[i, :] - self.x[j, :]) /
                    (2 * self.graphValue * self.graphValue))
                self.simMat[j, i] = self.simMat[i, j]

    def makeGraph(self):
        if self.graphWay == 'full':
            self.makeGraphFull()
        elif self.graphWay == 'epsilon':
            self.makeGraphEpsilon()
        elif self.graphWay == 'knearest':
            self.makeGraphKnearest()

    def makeGraphFull(self):
        self.adjMat = self.simMat

    def makeGraphEpsilon(self):
        pass

    def makeGraphKnearest(self):
        pass

    def calLaplaceMat(self):
        # 计算度矩阵degree matrix
        self.degreeMat = np.diag(np.sum(self.adjMat, axis=0))  # 按列求和，张成对角阵
        # 计算拉普拉斯矩阵
        self.laplaceMat = np.subtract(self.degreeMat, self.adjMat)

    def calEigenMat(self):
        eigValue, eigVector = lin.eig(self.laplaceMat)
        selIdx = np.argsort(np.abs(eigValue))  # 升序排列，返回从最小值开始的index
        self.eigMat = eigVector[:, selIdx[0:self.numCluster-1]]  # 一列一个特征向量，共有numCluster列，共有numInstance行

    def doKmeans(self):
        kmeansModel = KmeansClustering(numCluster=self.numCluster)
        self.clusterIdx = kmeansModel.fit(self.eigMat)

    @property
    def clusterIdx(self):
        if hasattr(self, "clusterIdx"):
            return self._clusterIdx
        else:
            return None

    @clusterIdx.setter
    def clusterIdx(self, clusterIdx):
        self._clusterIdx = clusterIdx






