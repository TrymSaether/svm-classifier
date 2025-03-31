import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.optimize import minimize
from functools import partial

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import the TestLinear function
from numerical_method.LinearExample import TestLinear

def prepare_data(w, b, n_A, n_B, margin, **kwargs):
    """
    Generates and prepares data for SVM training using TestLinear function.
    
    Returns:
    --------
    X : ndarray
        Feature matrix with all samples
    y : ndarray
        Labels for the samples (1 for class A, -1 for class B)
    list_A : ndarray
        Samples from class A only
    list_B : ndarray
        Samples from class B only
    """
    # Generate the data using TestLinear
    list_A, list_B = TestLinear(w, b, n_A, n_B, margin, **kwargs)
    
    # Combine the samples and create labels
    X = np.vstack((list_A, list_B))
    y = np.hstack((np.ones(n_A), -np.ones(n_B)))
    
    return X, y, list_A, list_B

class SVM:
    """
    Support Vector Machine implementation using quadratic programming optimization.
    """
    
    def __init__(self, kernel='linear', C=1.0, tol=1e-3, max_iter=100):
        """
        Initialize the SVM classifier.
        
        Parameters:
        -----------
        kernel : str or callable
            Kernel function to use. Can be 'linear', 'poly', 'rbf', or a callable function.
        C : float
            Regularization parameter. The strength of the regularization is inversely 
            proportional to C.
        tol : float
            Tolerance for stopping criterion.
        max_iter : int
            Maximum number of iterations for optimization.
        """
        self.kernel_type = kernel
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = None
        self.support_vectors_ = None
        self.support_vector_labels = None
        self.support_vector_indices = None
        self.b = None
        self.w = None
        self.X_train = None
        self.y_train = None
        self.kernel_fn = self._get_kernel_function(kernel)
        
    def _get_kernel_function(self, kernel_type):
        """
        Return the appropriate kernel function based on kernel_type.
        """
        if callable(kernel_type):
            return kernel_type
        
        if kernel_type == 'linear':
            return lambda x1, x2: np.dot(x1, x2.T)
        elif kernel_type == 'poly':
            return lambda x1, x2, degree=3, coef0=1: (np.dot(x1, x2.T) + coef0) ** degree
        elif kernel_type == 'rbf':
            return lambda x1, x2, gamma=0.1: np.exp(-gamma * np.linalg.norm(x1 - x2)**2)
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")
    
    def _compute_kernel_matrix(self, X):
        """
        Compute the kernel matrix for the dataset.
        """
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        
        # Compute the full kernel matrix
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel_fn(X[i].reshape(1, -1), X[j].reshape(1, -1))
                
        return K
    
    def _objective_function(self, alpha, P, q):
        """
        Objective function to minimize for SVM:
        (1/2) * alpha^T * P * alpha + q^T * alpha
        
        Parameters:
        -----------
        alpha : ndarray
            Lagrange multipliers
        P : ndarray
            Quadratic term matrix (kernel matrix * outer product of labels)
        q : ndarray
            Linear term vector (negative ones)
            
        Returns:
        --------
        float
            Value of the objective function
        """
        return 0.5 * np.dot(alpha, np.dot(P, alpha)) + np.dot(q, alpha)
    
    def _gradient(self, alpha, P, q):
        """
        Gradient of the objective function.
        """
        return np.dot(P, alpha) + q
    
    def fit(self, X, y):
        """
        Fit the SVM model according to the given training data.
        
        Parameters:
        -----------
        X : ndarray
            Training vectors of shape (n_samples, n_features)
        y : ndarray
            Target values of shape (n_samples,), values should be +1 and -1
            
        Returns:
        --------
        self : object
            Returns self.
        """
        n_samples, n_features = X.shape
        self.X_train = X
        self.y_train = y
        
        # Compute the kernel matrix
        K = self._compute_kernel_matrix(X)
        
        # Prepare matrices for the quadratic programming problem
        P = np.outer(y, y) * K
        q = -np.ones(n_samples)
        
        # Constraints: 0 <= alpha_i <= C and sum(alpha_i * y_i) = 0
        # We'll use scipy.optimize.minimize with bounds and constraints
        bounds = [(0, self.C) for _ in range(n_samples)]
        constraints = {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y), 'jac': lambda alpha: y}
        
        # Initial guess for alphas (all zeros)
        alpha0 = np.zeros(n_samples)
        
        # Solve the quadratic programming problem
        result = minimize(
            fun=self._objective_function,
            x0=alpha0,
            args=(P, q),
            method='SLSQP',
            jac=self._gradient,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iter}
        )
        
        # Extract the solution
        self.alpha = result.x
        
        # Find support vectors (samples with non-zero alpha)
        sv_threshold = 1e-5  # threshold to consider alpha as non-zero
        self.support_vector_indices = np.where(self.alpha > sv_threshold)[0]
        self.support_vectors_ = X[self.support_vector_indices]
        self.support_vector_labels = y[self.support_vector_indices]
        self.alpha = self.alpha[self.support_vector_indices]
        
        # Compute bias term b
        self.b = self._compute_bias()
        
        # If using linear kernel, compute w explicitly
        if self.kernel_type == 'linear':
            self._compute_w(n_features)
            
        return self
    
    def _compute_w(self, n_features):
        """
        Compute the weight vector w for linear SVM.
        """
        self.w = np.zeros(n_features)
        for i in range(len(self.alpha)):
            self.w += self.alpha[i] * self.support_vector_labels[i] * self.support_vectors_[i]
    
    def _compute_bias(self):
        """
        Compute the bias term using support vectors.
        """
        # For numerical stability, average over all support vectors
        b_sum = 0
        n_sv = len(self.support_vectors_)
        
        for i in range(n_sv):
            b_sum += self.support_vector_labels[i]
            b_sum -= np.sum([
                self.alpha[j] * self.support_vector_labels[j] * 
                self.kernel_fn(self.support_vectors_[j].reshape(1, -1), 
                             self.support_vectors_[i].reshape(1, -1))
                for j in range(n_sv)
            ])
            
        return b_sum / n_sv if n_sv > 0 else 0
    
    def predict(self, X):
        """
        Perform classification on samples in X.
        
        Parameters:
        -----------
        X : ndarray
            Input data of shape (n_samples, n_features)
            
        Returns:
        --------
        y_pred : ndarray
            Class labels for samples in X
        """
        decision_values = self.decision_function(X)
        return np.sign(decision_values)
    
    def decision_function(self, X):
        """
        Compute the decision function for the samples.
        
        Parameters:
        -----------
        X : ndarray
            Input data of shape (n_samples, n_features)
            
        Returns:
        --------
        ndarray
            Decision function values
        """
        if self.kernel_type == 'linear' and self.w is not None:
            # Fast computation for linear kernel
            return np.dot(X, self.w) + self.b
        else:
            # Kernel trick for non-linear kernels
            decision_values = np.zeros(X.shape[0])
            
            for i in range(X.shape[0]):
                s = 0
                for alpha_idx, sv_idx in enumerate(range(len(self.support_vectors_))):
                    s += self.alpha[alpha_idx] * self.support_vector_labels[alpha_idx] * \
                         self.kernel_fn(X[i].reshape(1, -1), 
                                      self.support_vectors_[sv_idx].reshape(1, -1))
                decision_values[i] = s
                
            return decision_values + self.b

def train_svm(X, y, C=1.0, kernel='linear'):
    """
    Trains a custom SVM model on the given data.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix
    y : ndarray
        Labels
    C : float
        Regularization parameter
    kernel : str
        Kernel type
        
    Returns:
    --------
    model : SVM
        Trained SVM model
    """
    model = SVM(kernel=kernel, C=C)
    model.fit(X, y)
    return model

def plot_svm_decision_boundary(model, X, y, list_A, list_B, w=None, b=None, title=None):
    """
    Visualizes the SVM decision boundary along with the data points.
    
    Parameters:
    -----------
    model : SVM
        Trained SVM model
    X : ndarray
        Feature matrix
    y : ndarray
        Labels
    list_A : ndarray
        Samples from class A
    list_B : ndarray
        Samples from class B
    w : ndarray, optional
        True normal vector for comparison
    b : float, optional
        True offset for comparison
    title : str, optional
        Plot title
    """
    # For 2D data only
    if X.shape[1] != 2:
        print("Visualization only supported for 2D data")
        return
    
    # Create a mesh grid for visualization
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Get predictions on the mesh grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.decision_function(grid_points)
    Z = Z.reshape(xx.shape)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.5,
                colors=('skyblue', 'white', 'pink'))
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], linestyles=['--', '-', '--'])
    
    # Plot the data points
    plt.scatter(list_A[:, 0], list_A[:, 1], c='red', label='Class A', edgecolors='k')
    plt.scatter(list_B[:, 0], list_B[:, 1], c='blue', label='Class B', edgecolors='k')
    
    # Highlight support vectors
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                s=100, facecolors='none', edgecolors='green', label='Support Vectors')
    
    # If true hyperplane parameters are provided, show the true decision boundary
    if w is not None and b is not None:
        if w.shape[0] == 2:  # Only for 2D
            slope = -w[0]/w[1]
            intercept = b/w[1]
            x_vals = np.array([x_min, x_max])
            y_vals = intercept + slope * x_vals
            plt.plot(x_vals, y_vals, 'g--', label='True Hyperplane')
    
    plt.title(title if title else 'SVM Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_svm(model, X, y):
    """
    Evaluates the SVM model on the given data.
    
    Returns:
    --------
    accuracy : float
        Accuracy of the model on the data
    """
    y_pred = model.predict(X)
    accuracy = np.mean(y_pred == y)
    print(f"Accuracy: {accuracy:.4f}")
    
    # For linear SVM, print the hyperplane parameters
    if model.kernel_type == 'linear':
        w = model.w
        b = model.b
        print(f"Learned hyperplane: w = {w}, b = {b}")
        return accuracy, w, b
    
    return accuracy, None, None

def main():
    # Set parameters for data generation
    d = 2  # dimensions
    w = np.array([1, 1])  # normal vector
    b = -2  # offset
    n_A = 100  # number of points in class A
    n_B = 100  # number of points in class B
    margin = 1  # margin between classes
    
    # Generate and prepare data
    print("Generating data...")
    X, y, list_A, list_B = prepare_data(w, b, n_A, n_B, margin, seed=42)
    
    # Train the SVM model
    print("Training SVM model...")
    model = train_svm(X, y, C=1.0)
    
    # Evaluate the model
    print("Evaluating model...")
    accuracy, learned_w, learned_b = evaluate_svm(model, X, y)
    
    # Plot the results
    print("Plotting results...")
    plot_svm_decision_boundary(
        model, X, y, list_A, list_B, 
        w=w, b=b, 
        title=f"SVM Decision Boundary (Accuracy: {accuracy:.4f})"
    )
    
    print("Done!")

if __name__ == "__main__":
    main()
