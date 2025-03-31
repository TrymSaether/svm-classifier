# utils.py
import numpy as np
from numpy.random import default_rng
def hinge_loss(w, b, X, y):
    """
    Compute total hinge loss = sum_i max(0, 1 - y_i*(w dot x_i + b))
    """
    margin = y * (X.dot(w) + b)
    return np.sum(np.maximum(0.0, 1 - margin))

def accuracy(y_true, y_pred):
    """
    Classification accuracy
    """
    return np.mean(y_true == y_pred)

def compute_primal_objective(w, b, X, y, C):
    """
    0.5 * ||w||^2 + C * hinge_loss
    """
    hl = hinge_loss(w, b, X, y)
    return 0.5 * np.sum(w*w) + C*hl

def get_support_vectors_primal(w, b, X, y):
    """
    Points with y_i*(w dot x_i + b) <= 1 are support vectors for primal.
    """
    margin_vals = y * (X.dot(w) + b)
    return np.where(margin_vals <= 1.0)[0]
def get_w(alpha, X, y):
    """
    Compute w from alpha.
    """
    return np.dot(alpha * y, X)
def get_support_vectors_dual(alpha, eps=1e-8):
    """
    Indices i with alpha_i > 0 are support vectors in the dual context.
    """
    return np.where(alpha > eps)[0]

def test_linear(w,b,n_A,n_B,margin,**kwargs):
    '''
    Parameters
    ----------
    w : non-zero vector
        normal vector defining a hyperplane
    b : real number
        offset of the hyperplane
    n_A : integer
        number of additional samples from class A
    n_B: integer
        number of additional samples from class B
    margin : positive real
        desired margin for the samples
        
    Optional Parameters
    -------------------
    seed : integer
        seed for the random number generator
        default value : 18
    sigma : positive real
        standard deviation for the normal distribution
        default value : 1.
    shape : positive real
        shape parameter for the Gamma distribution
        default value : 1.
    scale : positive real
        scale parameter for the Gamma distribution
        default value : 1.

    Returns
    -------
    list_A, list_B : lists of vectors
        list_A contains n_A vectors all lying on one side of the hyperplane H(w,-b).
        The distance to the hyperplane is margin + a sample of a Gamma distribution.
        In the plane normal to w, the points follow a normal distribution.
        One of the vectors acts as a support vector with precise margin gamma.
        list_B contains n_B vectors, produced in a similar way, lying on the
        opposite side of the hyperplane.

    '''
    
    # read out additional keyword arguments
    seed = kwargs.get("seed",18)
    shape = kwargs.get("shape",1.)
    scale = kwargs.get("scale",1.)
    sigma = kwargs.get("sigma",1.)
    
    # read out the number of dimensions
    d = w.size
    
    # rescale w to length 1
    norm_w = np.linalg.norm(w)
    w = w/norm_w
    b = b/norm_w

    # initialise a random number generator
    rng = default_rng(seed)
    list_A = []    
    for _ in range(n_A):

        vec = rng.normal(size=d,scale=sigma)
        dist = rng.gamma(shape,scale)
        # project vec onto w^\perp
        vec += -np.inner(vec,w)*w
        # add (dist+margin+b)*w to vec
        vec += (dist+margin-b)*w
        # append the vector vec to list_A
        list_A.append(vec)
        
    # initialise an empty list
    list_B = []    
    # draw samples for class A
    for _ in range(n_B):
        # draw n_B samples of a d-dimensional normal distribution
        vec = rng.normal(size=d,scale=sigma)
        # draw n_A samples of a Gamma distribution
        dist = rng.gamma(shape,scale)
        # project vec onto w^\perp
        vec += -np.inner(vec,w)*w
        # add -(dist+margin-b)*w to vec
        vec += (-b-dist-margin)*w
        # append the vector vec to list_B
        list_B.append(vec)
    
    # choose a random vector of each list and force it to be a support vector
    vec = rng.normal(size=d,scale=sigma)
    vec += -np.inner(vec,w)*w
    supp_A = rng.integers(0,n_A)
    list_A[supp_A] = vec+(margin-b)*w
    supp_B = rng.integers(0,n_B)
    list_B[supp_B] = vec+(-b-margin)*w
    X = np.vstack([list_A, list_B])
    y = np.concatenate([np.ones(n_A), -np.ones(n_B)])
    return X, y
