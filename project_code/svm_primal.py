# svm_primal.py
import numpy as np

class PrimalSVM:
    """
    Primal Soft-Margin SVM solved with subgradient descent on hinge loss.
    ------------------------------------------------------
    Objective:
      min_{w,b}  0.5 * ||w||^2  +  C * sum_i max(0, 1 - y_i * (w dot x_i + b))

    Parameters:
      C: Regularization parameter
      max_iter: Maximum number of subgradient updates
      tol: Gradient norm tolerance for stopping
      lr: Initial learning rate
      lr_decay: (Optional) decaying factor if we want diminishing step sizes
      verbose: Whether to print progress
      bb_steps: If True, enable Barzilai-Borwein step size adaptation
    """
    def __init__(self, C=1.0, max_iter=1000, tol=1e-4, lr=1e-2, 
                 lr_decay=0.0, verbose=False, bb_steps=False):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.lr = lr
        self.lr_decay = lr_decay
        self.verbose = verbose
        self.bb_steps = bb_steps
        
        # Learned parameters
        self.w = None
        self.b = 0.0
        
        # Keep track of objective history
        self.obj_history = []
    
    def fit(self, X, y):
        """
        Fit the model using subgradient descent on the hinge-loss objective.
        X: (M, d) data matrix
        y: (M,) labels in {-1, +1}
        """
        M, d = X.shape
        # Initialize
        self.w = np.zeros(d)
        self.b = 0.0
        
        lr = self.lr
        
        # For optional BB step
        prev_w = self.w.copy()
        prev_grad_w = np.zeros_like(self.w)
        prev_grad_b = 0.0
        
        for it in range(self.max_iter):
            margin = y * (X.dot(self.w) + self.b)  # shape (M,)
            
            # Identify which points are violating
            idx_violating = np.where(margin < 1)[0]
            
            # Gradient for w = w + C * sum(...) but subgradient sign is negative of that in the update
            grad_w = self.w.copy()  # derivative of 0.5||w||^2 is w
            grad_b = 0.0
            
            if len(idx_violating) > 0:
                # subgradient from hinge portion
                X_viol = X[idx_violating]
                y_viol = y[idx_violating]
                
                grad_w -= self.C * np.sum(y_viol[:, None] * X_viol, axis=0)
                grad_b -= self.C * np.sum(y_viol)
            
            # Record objective
            hinge_loss = np.sum(np.maximum(0.0, 1.0 - margin))
            obj_val = 0.5 * np.sum(self.w**2) + self.C * hinge_loss
            self.obj_history.append(obj_val)
            
            # Check stopping (norm of gradient)
            grad_norm = np.sqrt(np.sum(grad_w**2) + grad_b**2)
            if grad_norm < self.tol:
                if self.verbose:
                    print(f"Iteration {it}: grad norm {grad_norm:.4f} below tol -> stop.")
                break
            
            # Optionally compute Barzilai-Borwein step
            if self.bb_steps and it > 0:
                s_w = self.w - prev_w
                z_w = grad_w - prev_grad_w
                denom = np.dot(s_w, z_w)
                if abs(denom) > 1e-12:
                    lr_bb = np.dot(s_w, s_w) / denom
                    # clamp step size
                    lr = np.clip(lr_bb, 1e-8, 1e8)
            
            # Update w, b
            self.w = self.w - lr * grad_w
            self.b = self.b - lr * grad_b
            
            # store for next iteration
            prev_w = self.w.copy()
            prev_grad_w = grad_w.copy()
            prev_grad_b = grad_b
            
            # Optionally decay step size
            if self.lr_decay > 0.0:
                lr = self.lr / (1.0 + self.lr_decay * it)
            
            if self.verbose and it % 100 == 0:
                print(f"Iter {it}, Obj={obj_val:.4f}, GradNorm={grad_norm:.4f}, lr={lr:.4f}")
        
        if self.verbose:
            print("Finished subgradient descent.")
        return self
    
    def decision_function(self, X):
        """ Return w dot X + b (scores) """
        return X.dot(self.w) + self.b
    
    def predict(self, X):
        scores = self.decision_function(X)
        return np.sign(scores)
    
    def margin(self):
        """Margin = 1 / ||w|| if w != 0."""
        norm_w = np.linalg.norm(self.w)
        return 1.0 / norm_w if norm_w != 0 else np.inf
