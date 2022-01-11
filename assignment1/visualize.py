import numpy as np
filePath = "univariateData.dat"
file = open( filePath, 'r')
allData = np.loadtxt (file, delimiter=',')
X = np.matrix(allData[:,:-1])
y = np.matrix((allData[:,-1])).T
# get the number of in stances (n) and number of features (d)
n, d = X.shape
from test_linreg_univariate import plotData1D
plotData1D(X, y)