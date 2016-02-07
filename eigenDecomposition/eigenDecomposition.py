import numpy

A = numpy.array([[ 8,-1, 3,-1],
                 [-1, 6, 2, 0],
                 [ 3, 2, 9, 1],
                 [-1, 0, 1, 7]] , dtype = numpy.float32)

def givensRotation(M,c,s,p,q):
    tmp = M.copy()
    for i in xrange(len(M)):
        tmp[i,p] = M[i,p]*c - M[i,q]*s
        tmp[i,q] = M[i,p]*s + M[i,q]*c
    return tmp

def update(D,D_,eigenvectors,p,q):
    # Computing rotation angle that zeros out
    # 
    theta = (D_[q,q]-D_[p,p])/(2*D_[p,q])
    n = len(D)
    t = 1 / (numpy.abs(theta)+numpy.sqrt(numpy.power(theta,2)+1))
    t = t if theta>=0 else -t
    c = 1 / numpy.sqrt(numpy.power(t,2) + 1)
    s = c * t
    # Computing output of Givens matrix times
    # D_, the following code is just an optimization
    # of a matrix product (since just 2 rows and 2 columns
    # are affected by the transformations)
    D[p,q] = 0
    D[q,p] = 0
    D[p,p] = c**2*D_[p,p] + s**2*D_[q,q] - 2*c*s*D_[p,q]
    D[q,q] = s**2*D_[p,p] + c**2*D_[q,q] + 2*c*s*D_[p,q]
    for j in xrange(n):
        if(j!=p and j!=q):
            D[j,p] = c*D_[j,p]-s*D_[j,q]
            D[p,j] = D[j,p]
            D[j,q] = c*D_[j,q]+s*D_[j,p]
            D[q,j] = D[j,q]
    return givensRotation(eigenvectors,c,s,p,q)
    

D_ = A.copy()
D = A.copy()
eigenvectors = numpy.diag(numpy.ones(len(A)))
eps = 0.0
done = False
while(not done):
    done = True
    for i in xrange(len(A)):
        for j in xrange(len(A)):
            if j!=i:
                if numpy.abs(D_[i,j])>eps:
                    done = False
                    eigenvectors = update(D,D_,eigenvectors,i,j)
                    D_=D.copy()
print numpy.diag(D)
print eigenvectors
print numpy.dot(numpy.dot(eigenvectors,D),numpy.linalg.inv(eigenvectors))-A
    
        