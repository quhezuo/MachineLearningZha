# coding: utf-8

import numpy as np
from Clustering.SpectralClustering import SpectralClustering
import matplotlib.pyplot as plt

x = np.random.rand(200, 2)
model = SpectralClustering(graphWay='full', graphValue=0.3, numCluster=6)
model.fit(x)
clusterIdx = model.clusterIdx

plt.figure(1)
plt.xlabel('Position of X')
plt.ylabel('Position of Y')
plt.title('The Result of Spectral Clustering')
lineType = ['bo', 'rv', 'mv', 'gs', 'cp', 'yh']
assert len(lineType)==model.numCluster, 'Check the lineType'
for i, strLine in enumerate(lineType):
    plt.plot(x[clusterIdx==i, 0], x[clusterIdx==i, 1], strLine)

plt.show()





