# svm_primal.py
from svm_base import BaseSVM
import numpy as np

class PrimalSVM(BaseSVM):
    def __init__(self, C=1.0, max_iter=1000, tol=1e-12, lr=1e-5, 
                lr_decay=0.0, verbose=False, bb_steps=False):
        super().__init__(C=C, max_iter=max_iter, tol=tol)
        self.lr = lr
        self.lr_decay = lr_decay
        self.verbose = verbose
        self.bb_steps = bb_steps
        
        # Learned parameters
        self.w = None
        self.b = 0.0
        self.is_fitted = False 
    
    def __repr__(self):
        if not self.is_fitted:
            return "PrimalSVM(not fitted)"
        return f"PrimalSVM(C={self.C}, w={np.round(self.w, 4)}, b={self.b:.4f})"
        
    def fit(self, X, y):
        """
        Fit the model using subgradient descent on the hinge-loss objective.
        X: (M, d) data matrix
        y: (M,) labels in {-1, +1}
        """
        self.X = X
        self.y = y
        
        M, d = X.shape

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
            idx_violating = np.where(margin <= 1.0)[0]
            
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
                    print(f"Iteration {it}: grad norm {grad_norm:.4f} below tol -> stop., obj={obj_val:.4f}, w={np.round(self.w, 4)}, b={self.b:.4f}")
                break
            
            # Optionally compute Barzilai-Borwein step
            if self.bb_steps and it > 0:
                s_w = self.w - prev_w
                z_w = grad_w - prev_grad_w
                denom = np.dot(s_w, z_w)
                if denom > 0:
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
            
            if self.verbose and it % 1000 == 0:
                print(f"Iter {it}, Obj={obj_val:.4f}, GradNorm={grad_norm:.4f}, lr={lr:.4f}, w={np.round(self.w, 4)}, b={self.b:.8f}, grad_w={np.round(grad_w, 4)}, grad_b={grad_b:.4f}")
        
        if self.verbose:
            print("Finished subgradient descent.")
            print(f"Final objective value: {obj_val:.4f}, w={np.round(self.w, 4)}, b={self.b:.6f}")
        self.is_fitted = True
        return self
    
    def decision_function(self, X):
        """
        Calculate the decision function values for samples in X.
        
        The decision function for SVM gives the signed distance from the separating
        hyperplane to each sample. The sign of the decision function value determines 
        the predicted class.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to compute decision values for.
            
        Returns
        -------
        ndarray of shape (n_samples,)
            Decision function values for each sample.
            
        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call 'fit' first.")
        return X @ self.w + self.b
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict labels for.
            
        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels, either +1 or -1.
        """
        scores = self.decision_function(X)
        return np.sign(scores)
    
    def margin(self):
        """
        Calculate the margin of the SVM model.
        
        The margin is the distance between the separating hyperplane 
        and the closest data points (support vectors). For a linear SVM, 
        this equals 1/||w|| if w is non-zero.
        
        Returns
        -------
        float
            The margin value. Returns infinity if w is a zero vector.
        """
        norm_w = np.linalg.norm(self.w)
        return 1.0 / norm_w if norm_w != 0 else np.inf
        
    def get_support_vectors(self, eps=1e-3):
        if self.verbose:
            print(f'b: {self.b}, w: {self.w}')
        margin = self.y * (self.X.dot(self.w) + self.b)
        if self.verbose:
            print(f"Margin: {margin}")
        support_vector_indices = np.where(margin <= 1.0 + eps)[0]
        
        # Check support vectors from each class
        pos_sv = support_vector_indices[self.y[support_vector_indices] == 1.0]
        neg_sv = support_vector_indices[self.y[support_vector_indices] == -1.0]
        
        if self.verbose:
            print(f"Positive class support vectors: {len(pos_sv)}")
            print(f"Negative class support vectors: {len(neg_sv)}")
        return support_vector_indices
    def get_w(self):
        """
        Get the weight vector w.
        Returns
        -------
        ndarray
            The weight vector w.
        """
        return self.w
    
    def get_b(self):
        """
        Get the bias term b.
        Returns
        -------
        float
            The bias term b.
        """
        return self.b
    