# coding: utf-8


import numpy as np
from Common.BasicModel import BasicModel


# 决策树节点
class Node(object):
    def __init__(self, idxAttr=[], typeAttr='d', valSplit=[], sonNode={}, typeNode='null', label=None):
        self.idxAttr = idxAttr
        self.typeAttr = typeAttr
        self.valSplit = valSplit
        self.sonNode = sonNode
        self.typeNode = typeNode
        self.label = label
        self.labelScore = 0


# 决策树
class DecisionTree(BasicModel):
    def __init__(self, minInsInLeaf=5):
        self.minInsInLeaf = minInsInLeaf

    def fit(self, x, y, attrIsCont=[]):
        numAttr = x.shape[1]  # 属性数
        if not attrIsCont:  # 默认情况下所有属性为连续
            attrIsCont = np.ones(numAttr)
        attrIsAvailable = np.ones(numAttr)  # 所有属性均可选
        tree = self.BifurcateTree(x, y, attrIsCont, attrIsAvailable)
        tree.typeNode = 'root'  # 根节点
        self.tree = tree

    def predict(self, x):
        numInstance = x.shape[0]  # 输入的样本数
        predictLab = np.zeros(numInstance)
        for i in range(numInstance):
            predictLab[i], tmp = self.SearchOneInstance(x[i, :])
        return predictLab

    def SearchOneInstance(self, oneX):
        node = self.tree
        while node.typeNode != 'leaf':
            idxAttr = node.idxAttr  # 属性编号
            typeAttr = node.typeAttr  # 'd' 'c'
            valSplit = node.valSplit
            if typeAttr == 'c':
                if oneX[idxAttr] <= valSplit:
                    node = node.sonNode[0]
                else:
                    node = node.sonNode[1]
            elif typeAttr == 'd':
                if oneX[idxAttr] in node.sonNode:
                    node = node.sonNode[oneX[idxAttr]]
                else:
                    break
        return node.label, node.labelScore

    def BifurcateTree(self, x, y, attrIsCont, attrIsAvailable):
        thisNode = Node(idxAttr=[], typeAttr='', valSplit=[], sonNode={}, typeNode='null')
        numInstance = float(y.shape[0])
        numAttr = x.shape[1]
        # 统计不同类别样本的数量
        labelDict = self.CountCategoryNumber(y)  # 返回标签字典
        # 计算样本整体信息熵
        infoEntAll = self.CalInfoEntropy(labelDict)
        # 判断是否达到成为叶节点条件
        if len(labelDict) == 1:
            thisNode.typeNode = 'leaf'
            thisNode.label = labelDict.keys()[0]
            thisNode.labelScore = 1.0
            return thisNode
        elif infoEntAll < 1.0e-1 or np.sum(attrIsAvailable) == 0 or numInstance < self.minInsInLeaf:
            tmp = -1
            thisNode.typeNode = 'leaf'
            for i in labelDict:
                if labelDict[i] > tmp:
                    tmp = labelDict[i]
                    thisNode.label = i
                    thisNode.labelScore = labelDict[i] / numInstance
            return thisNode
        # 分别计算各划分值的信息增益和增益率
        splitInfo = np.zeros([0, 4])  # idxAttr, splitValue, infoGain, gainRatio
        for idx in np.arange(int(numAttr))[attrIsAvailable == 1]:  # idx 表示变量的index
            if attrIsCont[idx] == 1:  # 连续变量
                attrValue = np.unique(x[:, idx])
                for i in range(len(attrValue) - 1):  # 分割点比unique取值数量少1
                    splitValue = (attrValue[i]+attrValue[i+1]) / 2.0
                    lessSplitValue = x[:, idx] <= splitValue
                    moreSplitValue = x[:, idx] > splitValue
                    infoGain = infoEntAll - (np.sum(lessSplitValue) / numInstance) * \
                               self.CalInfoEntropy(self.CountCategoryNumber(y[lessSplitValue])) - \
                               (np.sum(moreSplitValue) / numInstance) * \
                               self.CalInfoEntropy(self.CountCategoryNumber(y[moreSplitValue]))
                    intrinsicVal = - (np.sum(lessSplitValue)/numInstance)*np.log2(np.sum(lessSplitValue)/numInstance) - \
                                   (np.sum(moreSplitValue)/numInstance)*np.log2(np.sum(moreSplitValue)/numInstance)
                    gainRatio = infoGain / intrinsicVal
                    splitInfo = np.row_stack((splitInfo, [idx, splitValue, infoGain, gainRatio]))
            else:  # 离散变量
                subsetDict = self.CountCategoryNumber(x[:, idx])
                infoGain = infoEntAll
                intrinsicVal = 0
                for iKey in subsetDict:
                    equal2KeyIdx = x[:, idx] == iKey
                    infoGain = infoGain - (np.sum(equal2KeyIdx) / numInstance) * \
                                          self.CalInfoEntropy(self.CountCategoryNumber(y[equal2KeyIdx]))
                    intrinsicVal = intrinsicVal - \
                                   (np.sum(equal2KeyIdx)/numInstance)*np.log2(np.sum(equal2KeyIdx)/numInstance)
                    gainRatio = infoGain / intrinsicVal
                    splitInfo = np.row_stack((splitInfo, [idx, None, infoGain, gainRatio]))
        # 选取最优划分属性和值
        infoGainBiggerIdx = splitInfo[:, 2] >= np.mean(splitInfo[:, 2])  # 信息增益大于平均水平
        selectRow = np.argmax(splitInfo[infoGainBiggerIdx, 3])
        bestAttrIdx, bestSplitVal = splitInfo[infoGainBiggerIdx, 0:2][selectRow, :]
        bestAttrIdx = int(bestAttrIdx)
        # 处理数据集用于生成子树
        if bestSplitVal is None:  # 离散变量
            thisNode.idxAttr = bestAttrIdx
            thisNode.typeAttr = 'd'
            thisNode.valSplit = bestSplitVal
            thisNode.typeNode = 'mid'
            attrIsAvailable[bestAttrIdx] = 0  # 离散变量后续不能再使用
            subsetDict = self.CountCategoryNumber(x[:, bestAttrIdx])
            for iKey in subsetDict:
                thisNode.sonNode[iKey] = self.BifurcateTree(x[x[:, bestAttrIdx] == iKey, :],
                                                     y[x[:, bestAttrIdx] == iKey], attrIsCont, attrIsAvailable)
        else:  # 连续变量
            thisNode.idxAttr = bestAttrIdx
            thisNode.typeAttr = 'c'
            thisNode.valSplit = bestSplitVal
            thisNode.typeNode = 'mid'
            thisNode.sonNode[0] = self.BifurcateTree(x[x[:, bestAttrIdx] <= bestSplitVal, :],
                                                 y[x[:, bestAttrIdx] <= bestSplitVal], attrIsCont, attrIsAvailable)
            thisNode.sonNode[1] = self.BifurcateTree(x[x[:, bestAttrIdx] > bestSplitVal, :],
                                                 y[x[:, bestAttrIdx] > bestSplitVal], attrIsCont, attrIsAvailable)
        # 返回结果
        return thisNode

    def CountCategoryNumber(self, y):
        labelDict = dict()
        if isinstance(y[0], (int, float, long)):
            for iL in y:
                if iL in labelDict:
                    labelDict[iL] += 1
                else:
                    labelDict[iL] = 1
        else:
            for iL in y:
                if iL[0] in labelDict:
                    labelDict[iL[0]] += 1
                else:
                    labelDict[iL[0]] = 1
        return labelDict

    def CalInfoEntropy(self, labelDict):
        numInstance = np.sum(labelDict.values())  # 样本总数
        infoEnt = 0
        for i in labelDict:
            tmp = float(labelDict[i]) / numInstance
            infoEnt = infoEnt - tmp * np.log2(tmp)
        return infoEnt

