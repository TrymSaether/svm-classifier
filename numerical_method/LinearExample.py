import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng


def TestLinear(w,b,n_A,n_B,margin,**kwargs):
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
    
    # initialise an empty list
    list_A = []    
    # draw samples for class A
    for _ in range(n_A):
        # draw n_A samples of a d-dimensional normal distribution
        vec = rng.normal(size=d,scale=sigma)
        # draw n_A samples of a Gamma distribution
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

    return(list_A,list_B)

class SoftMarginSVM:
    def __init__(self, C=1.0, max_epochs=1000, lr=0.001):
        """
        Parameters
        ----------
        C         : float
            Penalty parameter for misclassified (slack) points.
        max_epochs: int
            Number of passes (epochs) over the training set.
        lr        : float
            Fixed learning rate for subgradient steps.
        """
        self.C = C
        self.max_epochs = max_epochs
        self.lr = lr
        
        # Learned parameters:
        self.w_ = None  # shape (d,)
        self.b_ = None  # scalar
        # We can also track the history of the objective per epoch
        self.history_ = []

    def fit(self, X, y):
        """
        Fit a linear soft-margin SVM using subgradient descent.

        Parameters
        ----------
        X : np.ndarray, shape (d, M)
            Feature matrix with M samples in d dimensions
        y : np.ndarray, shape (M,)
            Labels in {+1, -1} for each sample

        Returns
        -------
        self : SoftMarginSVM
            Fitted model (for chaining / convenience)
        """
        d, M = X.shape

        # Initialize model parameters
        self.w_ = np.zeros(d)
        self.b_ = 0.0
        self.history_ = []

        for epoch in range(self.max_epochs):
            # Subgradient from the regularizer: derivative of 1/2||w||^2 is w
            grad_w = self.w_.copy()
            grad_b = 0.0

            # Compute margin: 1 - y_i(w^T x_i + b)
            margin = 1 - y * (self.w_ @ X + self.b_)
            # Indices where hinge loss is active
            active = (margin > 0)

            if np.any(active):
                # sum up subgradient contributions for all active samples
                grad_w -= self.C * np.sum((y[active] * X[:, active]), axis=1)
                grad_b -= self.C * np.sum(y[active])

            # Update model parameters
            self.w_ = self.w_ - self.lr * grad_w
            self.b_ = self.b_ - self.lr * grad_b

            # Evaluate objective for tracking
            hinge_vals = np.maximum(0.0, margin)
            obj = 0.5 * np.sum(self.w_**2) + self.C * np.sum(hinge_vals)
            self.history_.append(obj)

        return self

    def decision_function(self, X):
        """
        Compute signed distance from the decision boundary:
           f(x) = w^T x + b
        Parameters
        ----------
        X : np.ndarray, shape (d, M)
        Returns
        -------
        dist : np.ndarray, shape (M,)
            Signed distances
        """
        return self.w_ @ X + self.b_

    def predict(self, X):
        """
        Predict class labels +1 / -1 based on sign(w^T x + b)

        Parameters
        ----------
        X : np.ndarray, shape (d, M)

        Returns
        -------
        y_pred : np.ndarray, shape (M,)
            Predicted labels in {+1, -1}.
        """
        dist = self.decision_function(X)
        return np.where(dist >= 0.0, 1.0, -1.0)

def demo_svm_oop():
    # Generate data
    w_gt = np.array([1., 1.])
    b_gt = 1.
    nA, nB = 10, 8
    margin_gt = 0.5

    listA, listB = TestLinear(w_gt, b_gt, nA, nB, margin_gt, seed=42)
    # Convert to arrays
    A = np.array(listA).T  # shape (2, nA)
    B = np.array(listB).T  # shape (2, nB)

    # Combine data
    X = np.concatenate([A, B], axis=1)  # shape (2, nA+nB)
    yA = np.ones(nA)
    yB = -np.ones(nB)
    y = np.concatenate([yA, yB], axis=0)  # shape (nA+nB,)

    # Instantiate and fit the SVM
    svm = SoftMarginSVM(C=10.0, max_epochs=2000, lr=0.0005)
    svm.fit(X, y)

    print("Learned w =", svm.w_)
    print("Learned b =", svm.b_)

    # Evaluate on training set
    preds = svm.predict(X)
    accuracy = np.mean(preds == y)
    print("Training Accuracy = {:.2f}%".format(100*accuracy))

    # Plot data
    plt.figure()
    plt.scatter(A[0,:], A[1,:], color="red",  label="Class +1")
    plt.scatter(B[0,:], B[1,:], color="blue", label="Class -1")

    # Decision boundary
    x_min, x_max = X[0,:].min() - 1, X[0,:].max() + 1
    xx = np.linspace(x_min, x_max, 100)
    if abs(svm.w_[1]) > 1e-9:
        yy = -(svm.b_ + svm.w_[0]*xx)/svm.w_[1]
        plt.plot(xx, yy, 'g--', label="Decision boundary")

        # Margins: w^T x + b = ±1
        yy_plus = -(svm.b_ + svm.w_[0]*xx - 1)/svm.w_[1]
        yy_minus = -(svm.b_ + svm.w_[0]*xx + 1)/svm.w_[1]
        plt.plot(xx, yy_plus, 'k--', alpha=0.5)
        plt.plot(xx, yy_minus, 'k--', alpha=0.5)

    plt.title("Soft-Margin SVM (OOP, Subgradient)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.show()

    # Plot objective history
    plt.figure()
    plt.plot(svm.history_, label="Objective")
    plt.xlabel("Epoch")
    plt.ylabel("Objective value")
    plt.title("SVM Training Objective over Epochs")
    plt.legend()
    plt.show()

if __name__=="__main__":
    demo_svm_oop()