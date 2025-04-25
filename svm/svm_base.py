import numpy as np
from abc import ABC, abstractmethod

class BaseSVM(ABC):
    """
    Abstract base class for SVM implementations.
    
    This provides a common interface and shared functionality
    for both primal and dual SVM formulations.
    """
    
    def __init__(self, C=1.0, max_iter=1000, tol=1e-4, kernel=None):
        """
        Initialize base SVM parameters.
        
        Parameters:
        -----------
        C : float
            Regularization parameter
        max_iter : int
            Maximum number of iterations for the optimizer
        tol : float
            Convergence tolerance
        kernel : callable, optional
            Kernel function (for dual formulation)
        """
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.kernel = kernel
        self.obj_history = []
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X, y):
        """
        Fit the SVM model according to the given training data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training vectors
        y : array-like of shape (n_samples,)
            Target values (+1, -1)
            
        Returns:
        --------
        self : object
        """
        pass
    
    @abstractmethod
    def decision_function(self, X):
        """
        Compute the decision function of the model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input samples
            
        Returns:
        --------
        array-like of shape (n_samples,)
            Decision function values
        """
        pass
    
    def predict(self, X):
        """
        Perform classification on samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input samples
            
        Returns:
        --------
        array-like of shape (n_samples,)
            Class labels (+1 or -1)
        """
        return np.sign(self.decision_function(X))
    
    def score(self, X, y):
        """
        Return the accuracy on the given test data and labels.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True labels for X
            
        Returns:
        --------
        float
            Accuracy score
        """
        from utils import accuracy
        return accuracy(self.predict(X), y)
    
    def get_support_vectors(self):
        """
        Get the indices of support vectors.
        
        Returns:
        --------
        array-like
            Indices of support vectors
        """
        pass  # To be implemented in subclasses