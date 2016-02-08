#########################################################
#
#   Description: Implementation of the Jacobi algorithm
#   used to find eigenvalues and eigenvectors of a
#   simmetric matrix.
#   Additional Sources: 
#   http://mathfaculty.fullerton.edu/mathews//n2003/jacobimethod/jacobimethodproof.pdf
#
#########################################################

import numpy

A = numpy.array([[ 8,-1, 3,-1],
                 [-1, 6, 2, 0],
                 [ 3, 2, 9, 1],
                 [-1, 0, 1, 7]] , dtype = numpy.float32)

class GivensMatrix(object):
    def __init__(self,c,s,p,q):
        self._c = c
        self._s = s
        self._p = p
        self._q = q
    def inplaceTranspose(self):
        self._s = -self._s
    def inplaceLeftDot(self,X):
        for i in xrange(len(X)):
            tmpP = X[self._p,i]*self._c + X[self._q,i]*self._s
            tmpQ = X[self._q,i]*self._c - X[self._p,i]*self._s
            X[self._p,i] = tmpP
            X[self._q,i] = tmpQ
    def inplaceRightDot(self,X):
        for i in xrange(len(X)):
            tmpP = X[i,self._p]*self._c - X[i,self._q]*self._s
            tmpQ = X[i,self._p]*self._s + X[i,self._q]*self._c
            X[i,self._p] = tmpP
            X[i,self._q] = tmpQ
        
        
    
class JacobiEigendecomposition(object):
    def __init__(self, eps = 1e-6):
        self._eps = eps
        
    def _getGivensMatrix(self,M,p,q):
        """ 
        Computes the Givens Transform that cancels out M[p,q] and M[q,p]
        """ 
        theta = (M[q,q] - M[p,p])/(2*M[p,q])
        n = len(M)
        t = 1 / (numpy.abs(theta)+numpy.sqrt(numpy.power(theta,2)+1))
        t = t if theta>=0 else -t
        c = 1 / numpy.sqrt(numpy.power(t,2) + 1)
        s = c * t
        return GivensMatrix(c,s,p,q)
    
    def _update(self,D,eigenvectors,p,q):
        
        gm = self._getGivensMatrix(D,p,q)
        gm.inplaceTranspose()
        gm.inplaceLeftDot(D)
        gm.inplaceTranspose()
        gm.inplaceRightDot(D)
        gm.inplaceRightDot(eigenvectors)
        
    def fit(self, X):
        D = X.copy()
        eigenvectors = numpy.diag(numpy.ones(len(X)))
        done = False
        while(not done):
            done = True
            for i in xrange(len(A)):
                for j in xrange(len(A)):
                    if j!=i:
                        if numpy.abs(D[i,j])>self._eps:
                            done = False
                            self._update(D,eigenvectors,i,j)
        return eigenvectors,numpy.diag(D)
    

model = JacobiEigendecomposition()
eigenvectors, eigenvalues = model.fit(A)
print "_____________________________________________________________________"
print "Eigenvalues (V)"
print eigenvalues
print "_____________________________________________________________________"
print "Eigenvectors (A)"
print eigenvectors
print "_____________________________________________________________________"
print "Matrix Reconstruction Fidelity test ( A*diag(V)*inverse(A) - A = 0 ):"
print numpy.dot(numpy.dot(eigenvectors,numpy.diag(eigenvalues)),numpy.linalg.inv(eigenvectors))-A
print "_____________________________________________________________________"