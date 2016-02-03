#########################################################
#
#   Author: Matteo Presutto
#   Description: Implementation of a simple dual ridge
#   regressor with L2 regularization
#
#########################################################

import numpy

class Kernel(object):
    def __init__(self):
        pass
    
    def kernel(self, x1, x2, **kwargs):
        pass
    
    def kernelInnerProduct(self, x1, x2, **kwargs):
        tmp = numpy.zeros((x1.shape[0],x2.shape[1]),dtype=numpy.float32)
        for i,row in enumerate(x1):
            for j,col in enumerate(x2.T):
                tmp[i,j] = self.kernel(row,col,sigma=kwargs['sigma'])
        return tmp
    
    def grahamMatrix(self, X, **kwargs):
        """
        Computes the graham matrix using the function <kernel(x1,x2)>
        """
        result = self.kernelInnerProduct(X, X.T, sigma=kwargs['sigma'])
        return result
    

class ExponentialKernel(Kernel):
    def __init__(self):
        pass
    
    def kernel(self, x1, x2, **kwargs):
        tmp = x1-x2
        tmp = -kwargs['sigma']*numpy.linalg.norm(tmp)**2
        tmp = numpy.exp(tmp)
        return tmp

class DualRidgeRegressor(object):
    # input:
    # sigma -> Exponential kernel variable
    # l     -> l2 regularization hyperparameter
    def __init__(self, sigma, l, kernel = ExponentialKernel()):
        self._sigma = sigma
        self._l = l
        self._kernel = kernel
        
    def _getAlphas(self, X, y):
        grMatrix = self._kernel.grahamMatrix(X,sigma=self._sigma)
        lambdaRegularizer = numpy.diag(numpy.ones(X.shape[0])*self._l)
        alphas = numpy.dot(y,numpy.linalg.inv(grMatrix-lambdaRegularizer))
        return alphas
    
    def fit(self, X, y):
        self._alphas = self._getAlphas(X, y)
        self._X = X
        
    def predict(self, x):
        tmp = self._kernel.kernelInnerProduct(x[:,None].T, self._X.T, sigma=self._sigma)
        result = numpy.dot(tmp, self._alphas)
        return result
    
################ FAKE ARTIFICIAL DATA ##################
Xtrain = numpy.array([[1,2,3],
                    [4,5,6],
                    [7,8,9],
                    [10,11,12]], dtype=numpy.float32)
ytrain = numpy.array([1,-1,2,3], dtype=numpy.float32)

sigma = 0.1     # rbf constant
l = 0.001       # l2 regularizer constant

model = DualRidgeRegressor(sigma, l, kernel = ExponentialKernel())
model.fit(Xtrain, ytrain)

################ TESTING IMPLEMENTATION ################
predictions = model.predict(Xtrain.T)
print "Expected results:\t\t", ytrain
print "Predicted results:\t\t", predictions
print "Mean Absolute Difference:\t", numpy.abs(ytrain-predictions).mean()
