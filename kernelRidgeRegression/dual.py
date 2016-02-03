import numpy

def kernel(x1,x2,**kwargs):
    tmp = x1-x2
    tmp = -kwargs['sigma']*numpy.linalg.norm(tmp)**2
    tmp = numpy.exp(tmp)
    return tmp

def kernelDot(x1,x2,**kwargs):
    tmp = numpy.zeros((x1.shape[0],x2.shape[1]),dtype=numpy.float32)
    for i,row in enumerate(x1):
        for j,col in enumerate(x2.T):
            tmp[i,j] = kernel(row,col,sigma=kwargs['sigma'])
    return tmp

def grahamMatrix(X,**kwargs):
    """
    Computes the graham matrix using the function <kernel(x1,x2)>
    """
    result = kernelDot(X,X.T,sigma=kwargs['sigma'])
    return result

def getAlphas(X,y,**kwargs):
    grMatrix = grahamMatrix(X,sigma=kwargs['sigma'])
    lambdaRegularizer = numpy.diag(numpy.ones(X.shape[0])*kwargs['l'])
    alphas = numpy.dot(y,numpy.linalg.inv(grMatrix-lambdaRegularizer))
    return alphas
    
def predict(x,X,alphas,**kwargs):
    tmp=kernelDot(x[:,None].T,X.T,sigma=kwargs['sigma'])
    result = numpy.dot(tmp,alphas)
    return result
    
################ FAKE ARTIFICIAL DATA ##################
Xtrain = numpy.array([[1,2,3],
                    [4,5,6],
                    [7,8,9],
                    [10,11,12]], dtype=numpy.float32)
ytrain = numpy.array([1,-1,2,3], dtype=numpy.float32)
xtest = numpy.array([1,2,3], dtype=numpy.float32)

###################### PARAMETERS ######################
sigma = 0.1 # rbf constant
l = 0.001     # l2 regularizer constant

alphas = getAlphas(Xtrain, ytrain, sigma=sigma, l=l)

print predict(xtest, Xtrain, alphas, sigma=sigma)
