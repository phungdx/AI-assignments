import numpy as np
def mapFeature(x1, x2):
    '''
    Maps the two input features to quadratic features.
        
    Returns a new feature array with d features, comprising of
        X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, ... up to the 6th power polynomial
        
    Arguments:
        X1 is an n-by-1 column matrix
        X2 is an n-by-1 column matrix
    Returns:
        an n-by-d matrix, where each row represents the new features of the corresponding instance
    '''
    n = len(x1)
    X = np.zeros((n,27))
    idx = 0
    for i in range(1,7):
        X[:,idx] = np.power(x1,i)
        idx += 1

    for i in range(1,7):
        X[:,idx] = np.power(x2,i)
        idx += 1

    for i in range(1,7):
        for j in range(1,7):
            if i+j <= 6:
                X[:,idx] = np.power(x1,i) + np.power(x2,j)
                idx += 1

    return X




