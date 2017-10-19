import numpy as np

from jive.lazymatpy.interface import aslinearoperator, LinearOperator, IdentityOperator

class Ones(LinearOperator):
    def __init__(self, shape):
        super(Ones, self).__init__(dtype=None, shape=shape)

    def _matvec(self, x):
         return np.repeat(x.sum(), self.shape[0])

    def _rmatvec(self, x):
        return np.repeat(x.sum(), self.shape[1])

class OnesOuterVec(LinearOperator):
    def __init__(self, num_ones, vec):
        
        vec = np.asarray(vec)

        if (vec.ndim == 1) or (vec.shape[1] == 1):
            self.vec = vec

        elif (vec.ndim == 2) and (vec.shape[0] == 1):
            self.vec = vec.reshape(-1)
            
        else:
            raise ValueError('vec must be a vector')

        shape = (num_ones, self.vec.shape[0])
        dtype = self.vec.dtype

        super(OnesOuterVec, self).__init__(dtype=dtype, shape=shape)

    def _matvec(self, x):
         return np.repeat(self.vec.T.dot(x), self.shape[0])

    def _rmatvec(self, x):
        return self.vec * x.sum()