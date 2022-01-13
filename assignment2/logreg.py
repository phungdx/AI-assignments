'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''


import numpy as np
import matplotlib.pyplot as plt
class LogisticRegression:

    def __init__(self, alpha = 0.1, regLambda=0.001, epsilon=0.0001, maxNumIters = 10000):
        '''
        Constructor
        '''
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters

    

    def computeCost(self, theta, X, y, regLambda):
        '''
        Computes the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost  ** make certain you're not returning a 1 x 1 matrix! **
        '''
        n,d = X.shape
        h = self.sigmoid(X.dot(theta))
        cost = -np.mean((y*np.log(h) + (1-y)*np.log(1-h))) + regLambda/2 * np.linalg.norm(theta[1:,0]) ** 2
        return cost

    
    
    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an d-dimensional vector
        '''
        h = self.sigmoid(X.dot(theta))
        dl_wrt_theta = np.zeros((theta.shape))
        dl_wrt_theta = 1/len(X)*X.T.dot(h-y) + regLambda*theta
        dl_wrt_theta[0,0] = 1/len(X)*np.sum(h-y)

        return dl_wrt_theta
    


    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        '''
        n,d = X.shape
        X = np.c_[np.ones((n,1)), X]

        self.theta = np.random.randn(d+1,1)
        self.cost = []

        for _ in range(self.maxNumIters):
            cost = self.computeCost(self.theta,X,y,self.regLambda)
            new_theta = self.theta -  self.alpha * self.computeGradient(self.theta,X,y,self.regLambda)
            if np.linalg.norm(new_theta-self.theta) <= self.epsilon:
                break
            self.theta = new_theta
            self.cost.append(cost)


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions
        '''
        n,d = X.shape
        X = np.c_[np.ones((n,1)), X]
        predict = self.sigmoid(X.dot(self.theta))
        return np.round(predict)



    def sigmoid(self, Z):
        return 1.0/(1+np.exp(-Z))


    def plot_loss(self):
        plt.plot(self.cost)
        plt.title('Loss log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()