import numpy as np
from scipy.sparse import dia_matrix

from jive.lazymatpy.interface import aslinearoperator, LinearOperator, IdentityOperator
from jive.lazymatpy.templates.ones import OnesOuterVec


def col_mean_centered(S):
    """ S - 1m^T where m = col means of S """
    m = np.asarray(S.mean(axis=0)).reshape(-1)
    return aslinearoperator(S) - OnesOuterVec(num_ones=S.shape[0],
                                              vec=m)

def col_proj(M, U):
    """ UU^TM """
    return aslinearoperator(U) * aslinearoperator(U.T) * aslinearoperator(M)

def col_proj_orthog(M, U):
    """ (I - UU^T)M """
    return  (IdentityOperator(shape=(M.shape[0], M.shape[0]), dtype=M.dtype) - aslinearoperator(U) * aslinearoperator(U.T)) * aslinearoperator(M)


def svd_residual(X, U, D, Vt):
    """ X - UDVt """
    Xlo = aslinearoperator(X)
    Ulo = aslinearoperator(U)
    DVtlo = aslinearoperator(dia_matrix(np.diag(D)).dot(Vt))

    return Xlo - Ulo.dot(DVtlo)