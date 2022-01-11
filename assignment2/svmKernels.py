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
    # X1=np.array(X1.reshape(1,-1))
    # X2=np.array(X2.reshape(1,-1))
    norm = np.linalg.norm(X1) * np.linalg.norm(X2)
    return np.dot(X1, X2.T)/norm

