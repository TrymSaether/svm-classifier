# svm_dual.py
from projections import project_alpha
from svm_base import BaseSVM
import numpy as np
from kernels import linear_kernel


# class DualSVM:
#     """
#     Dual Soft-Margin SVM solved with Projected Gradient Descent.
# 
#     Dual objective:
#        min_{alpha}  0.5 alpha^T (Y G Y) alpha  -  1^T alpha
#        subject to:   sum_i y_i alpha_i = 0
#                      0 <= alpha_i <= C
#     
#     We can pass in a kernel function K(x_i, x_j). For linear SVM, K is just the dot product.
#     """
#     def __init__(self, C=1.0, kernel_func=None, max_iter=1000, tol=1e-6, 
#                  use_line_search=True, bb_steps=True, verbose=False):
#         """
#         Args:
#           C: Regularization parameter
#           kernel_func: a function K(x, z) that returns the scalar kernel value
#           max_iter: Maximum PGD iterations
#           tol: Tolerance for alpha updates
#           use_line_search: If True, do a line search when objective fails to decrease
#           bb_steps: If True, use Barzilai-Borwein step size
#           verbose: Print progress info
#         """
#        self.C = C
#        self.kernel_func = kernel_func
#        self.max_iter = max_iter
#        self.tol = tol
#        self.use_line_search = use_line_search
#        self.bb_steps = bb_steps

class DualSVM(BaseSVM):
    def __init__(self, C=1.0, max_iter=1000, tol=1e-4, kernel=linear_kernel, use_line_search=True, bb_steps=True, verbose=False):
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
        X: shape (M, d)
        y: shape (M,), entries in {-1, +1}
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
        diag_y = self.y
        # shape (M,M)
        Q = np.einsum('i,ij,j->ij', diag_y, self.G, diag_y, optimize='greedy')
        
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
                    # clamp
                    step = np.clip(step_bb, 1e-12, 1e12)
                else:
                    step = 1e-2
                
                prev_alpha = alpha.copy()
                prev_grad = g_new
            
            if self.verbose and k % 50 == 0:
                print(f"Iter={k}, Obj={new_obj:.6f}, Step={step:.4g}, ||alpha_diff||={np.linalg.norm(alpha_diff):.3e}")
        
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

