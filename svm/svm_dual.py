# svm_dual.py
from svm_base import BaseSVM
import numpy as np
from kernels import linear_kernel
from utils import project_alpha

class DualSVM(BaseSVM):
    """
    Support Vector Machine solver using the dual formulation with Projected Gradient Descent.
    
    This implementation solves the dual SVM optimization problem:
        min_α 0.5 * α^T Q α - 1^T α
        subject to: 0 ≤ α_i ≤ C for all i
                  y^T α = 0
                  
    Where Q_ij = y_i y_j K(x_i, x_j) and K is the kernel function.
    
    The solution yields the weights through the relation w = Σ_i α_i y_i x_i
    and supports both linear and non-linear kernels.
    """
    def __init__(self, C=1.0, max_iter=1000, tol=1e-4, kernel=linear_kernel, use_line_search=True, bb_steps=True, verbose=False):
        """
        Initialize the Dual SVM.
        
        Parameters
        ----------
        C : float, default=1.0
            Regularization parameter. The strength of the regularization is
            inversely proportional to C. Must be strictly positive.
        max_iter : int, default=1000
            Maximum number of iterations for the optimization algorithm.
        tol : float, default=1e-4
            Tolerance for stopping criterion.
        kernel : callable, default=linear_kernel
            Kernel function to use. Default is the linear kernel.
        use_line_search : bool, default=True
            Whether to use line search for step size selection.
        bb_steps : bool, default=True
            Whether to use Barzilai-Borwein step size selection.
        verbose : bool, default=False
            Whether to print progress during optimization.
        """
        super().__init__(C=C, max_iter=max_iter, tol=tol, kernel=kernel)
        self.use_line_search = use_line_search
        self.bb_steps = bb_steps
        self.verbose = verbose
        
        # Learned parameters:
        self.alpha = None
        self.b = 0.0
        self.X = None
        self.y = None
        self.G = None  # Gram matrix, or None if large-scale
        
        self.obj_history = []
        
    def fit(self, X, y):
        """
        Fit the Dual SVM using Projected Gradient Descent.
        
        This method solves the dual optimization problem using projected gradient descent
        with optional line search and Barzilai-Borwein step sizes. The algorithm performs
        the following steps:
        1. Computes the Gram matrix using the specified kernel function
        2. Initializes alpha to zeros
        3. Iteratively updates alpha using gradient descent with projection
        4. Computes the bias term b from support vectors
        
        Parameters
        ----------
        X : array-like of shape (M, d)
            Training data, where M is the number of samples and d is the number of features.
        y : array-like of shape (M,)
            Target values, must contain only {-1, +1} values.
            
        Returns
        -------
        self : object
            Returns self.
        """
        self.X = X
        self.y = y.astype(float)
        self.X_train = X
        self.y_train = y
        M, d = X.shape
        
        # Build Gram matrix G_ij = K(x_i, x_j)
        # For linear case, kernel_func(x_i, x_j) = x_i dot x_j
        self.G = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                self.G[i, j] = self.kernel(X[i], X[j])
        
        # Precompute Y G Y
        # We'll store Q = Y G Y so that gradient is Q alpha - 1
        
        diag_Y = self.y
        # shape (M,M)
        Q = np.einsum('i,ij,j->ij', diag_Y, self.G, diag_Y, optimize='greedy')
        
        # Initialize alpha = 0 which is feasible if y^T alpha=0
        alpha = np.zeros(M)
        
        # Step length initialization
        step = 1e-2
        
        # Evaluate objective
        def dual_objective(a):
            # f(a) = 0.5 a^T Q a - sum(a)
            return 0.5 * np.dot(a, Q.dot(a)) - np.sum(a)
        
        # Gradient
        def grad(a):
            # grad f(a) = Q a - 1
            return Q.dot(a) - np.ones(M)
        
        # Projected gradient descent
        current_obj = dual_objective(alpha)
        self.obj_history = [current_obj]
        
        # For BB step
        prev_alpha = alpha.copy()
        prev_grad = grad(alpha)
        
        # For line search references
        fref = np.inf
        fbest = current_obj
        candidate_obj = current_obj
        no_improve_count = 0
        L = 10  # how many consecutive iters we can go w/o improvement before updating fref
        
        for k in range(self.max_iter):
            g = grad(alpha)
            
            # unconstrained update
            beta = alpha - step * g
            # project onto feasible set
            alpha_new = project_alpha(beta, self.y, self.C)
            
            # Optional line search if f(alpha_new) > fref or first iteration
            new_obj = dual_objective(alpha_new)
            
            if (self.use_line_search and (new_obj > fref or k == 0)):
                # exact line search for quadratic
                d = alpha_new - alpha  # search direction
                # we want to min_{0<=theta<=1} f(alpha + theta d)
                # f(a + t d) = 0.5 (a+t d)^T Q (a+t d) - sum(a+t d)
                # This is a piecewise function once we include projection. 
                # But alpha_new is already projected. 
                # We'll do a standard line search with direct formula for 
                # arg min_{theta in [0,1]} of that quadratic. 
                # But we must keep feasibility. Let's attempt the direct approach:
                
                # We'll compute the derivative wrt theta and set =0:
                # grad wrt theta is d^T Q (a + theta d) - sum(d).
                # => d^T Q a + theta d^T Q d - d^T 1 = 0 => solve for theta
                # Then clamp to [0,1].
                
                Qa = Q.dot(alpha)
                Qd = Q.dot(d)
                num = - np.dot(d, Qa) + np.sum(d)
                den = np.dot(d, Qd)
                
                if abs(den) > 1e-12:
                    theta_star = num / den
                    theta_star = np.clip(theta_star, 0.0, 1.0)
                else:
                    theta_star = 1.0  # fallback if no curvature in direction
                
                alpha_mid = alpha + theta_star*d
                alpha_mid = project_alpha(alpha_mid, self.y, self.C)
                mid_obj = dual_objective(alpha_mid)
                
                # choose best among alpha_new, alpha_mid
                if mid_obj < new_obj:
                    alpha_new = alpha_mid
                    new_obj = mid_obj
            
            # Evaluate new objective
            if new_obj < fbest:
                fbest = new_obj
                candidate_obj = new_obj
                no_improve_count = 0
            else:
                candidate_obj = max(candidate_obj, new_obj)
                no_improve_count += 1
            
            if no_improve_count == L:
                fref = candidate_obj
                candidate_obj = new_obj
                no_improve_count = 0
            
            # Check for convergence
            if np.linalg.norm(alpha_new - alpha) < self.tol:
                alpha = alpha_new
                self.obj_history.append(new_obj)
                break
            
            alpha_diff = alpha_new - alpha
            alpha = alpha_new
            self.obj_history.append(new_obj)
            
            # Update step length if BB steps are used
            if self.bb_steps:
                g_new = grad(alpha)
                s = alpha - prev_alpha
                yv = g_new - prev_grad
                denom = np.dot(s, yv)
                if abs(denom) > 1e-12:
                    step_bb = np.dot(s, s) / denom
                    step = np.clip(step_bb, 1e-5, 1e5)
                else:
                    step = 1e-2
                
                prev_alpha = alpha.copy()
                prev_grad = g_new
            
            if self.verbose and k % 100 == 0:
                print(f"Iteration {k}: obj={new_obj:.4f}, ||g||={np.linalg.norm(g):.4f}, step={step:.4f}")
        # final alpha
        self.alpha = alpha
        
        # compute b from any i with 0 < alpha_i < C
        # For numerical stability, average over them:
        idx_margin = np.where((alpha > 1e-10) & (alpha < self.C - 1e-10))[0]
        if len(idx_margin) == 0:
            # fallback if no alpha in (0,C)
            # use any alpha_i > 0
            idx_pos = np.where(alpha > 1e-10)[0]
            if len(idx_pos) == 0:
                self.b = 0.0
            else:
                i0 = idx_pos[0]
                sum_ = 0.0
                # For linear or kernel
                w_xi = 0.0
                for j in range(M):
                    w_xi += alpha[j]*y[j]*self.kernel(X[j], X[i0])
                self.b = y[i0] - w_xi
        else:
            b_vals = []
            for i0 in idx_margin:
                w_xi = 0.0
                for j in range(M):
                    w_xi += alpha[j]*y[j]*self.kernel(X[j], X[i0])
                b_vals.append(y[i0] - w_xi)
            self.b = np.mean(b_vals)
        self.is_fitted = True
        return self
    
    def decision_function(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call 'fit' first.")
        # Kernel calculation between X and support vectors
        K = np.array([[self.kernel(x, self.X_train[i]) 
                      for i in range(len(self.X_train))] 
                      for x in X])
        return np.sum(self.alpha * self.y_train * K, axis=1) + self.b
    
    def predict(self, Xtest):
        """ Sign of decision_function. """
        return np.sign(self.decision_function(Xtest))
    
    def get_alpha(self):
        return self.alpha
    
    def get_b(self):
        return self.b
    
    def recover_w(self):
        """
        Recover w from alpha.
        For linear SVM, w = sum_i alpha_i y_i x_i
        """
        w = np.zeros(self.X.shape[1])
        for i in range(len(self.alpha)):
            w += self.alpha[i] * self.y[i] * self.X[i]
        return w

    def get_support_vectors(self):
            from utils import get_support_vectors_dual
            return get_support_vectors_dual(self.alpha)
        
    def get_margin(self):
        """
        Compute the margin of the SVM.
        """
        return 1.0 / np.linalg.norm(self.recover_w())

