"""
Custom SVM Kernels

Author: Eric Eaton, 2014

"""

import numpy as np

_polyDegree = 2
_gaussSigma = 1


def myPolynomialKernel(X1, X2):
    '''
        Arguments:  
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''

    return np.power((X1.dot(X2.T)+1), _polyDegree)



def myGaussianKernel(X1, X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    X = np.linalg.norm(X1[:, None, :] - X2[None, :, :], axis=-1)
    return np.exp(-1/(2*(_gaussSigma**2)) * X)




def myCosineSimilarityKernel(X1,X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    kernel = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            norm_xi, norm_xj = (np.linalg.norm(v) for v in (X1[i], X2[j]))
            XiT_Xj= np.dot(X1[i], X2[j])
            kernel[i, j] = XiT_Xj/(norm_xi*norm_xj)
    return kernel

