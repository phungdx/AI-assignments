import numpy as np
from sklearn.model_selection import LeaveOneOut
X = np.array([[1], [3],[6]])
print(np.sum(X[1:,:]**2))