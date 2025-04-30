# utils.py
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
# projections.py
import numpy as np

def project_alpha(beta, y, C):
    """
    Project beta onto the set { alpha : 0 <= alpha_i <= C, sum_i y_i alpha_i = 0 }.
    We find alpha = clamp(beta + lambda*y, 0, C) for some lambda such that sum_i y_i alpha_i = 0.
    Uses a bisection on lambda.
    
    beta: (M,) unconstrained update
    y: (M,), in {-1, +1}
    C: box constraint
    """
    # First do a trivial clamp w/o equality constraint
    alpha0 = np.clip(beta, 0, C)
    sum0 = np.dot(y, alpha0)
    
    # If sum0 == 0 => alpha0 is already feasible
    if abs(sum0) < 1e-14:
        return alpha0
    
    # We need to find lambda s.t. sum_i y_i clamp(beta_i + lambda*y_i,0,C) = 0
    # define g(lambda) = sum_i y_i clamp(beta_i + lambda*y_i,0,C).
    # We do bisection to find lambda that makes g(lambda) = 0.
    
    # Step 1: find a bracket [lamL, lamR] where g has different signs
    lamL = 0.0
    lamR = 0.0
    # We guess an initial delta
    delta = 1.0
    
    # function to compute g(lambda)
    def g(lam):
        return np.dot(y, np.clip(beta + lam*y, 0, C))
    
    # We want g(lam) to cross 0
    g0 = g(0.0)  # = sum0
    if g0 > 0:
        # we want lam to reduce the sum => let's move lam negatively
        lamL = 0.0
        lamR = 0.0
        while g(lamR) > 0:  # push lamR negative
            lamR -= delta
            delta *= 2
    else:
        # we want lam to increase the sum => let's move lam positively
        lamL = 0.0
        lamR = 0.0
        while g(lamR) < 0:
            lamR += delta
            delta *= 2
    
    # Now we have g(lamL)*g(lamR) < 0 hopefully
    # We'll ensure lamL < lamR
    if lamR < lamL:
        lamL, lamR = lamR, lamL
    
    gL = g(lamL)
    gR = g(lamR)
    # We expect gL*gR <= 0
    # Bisection
    for _ in range(50):  # 50 iters is plenty for double precision
        lamM = 0.5*(lamL + lamR)
        valM = g(lamM)
        if abs(valM) < 1e-12:
            # good enough
            return np.clip(beta + lamM*y, 0, C)
        if valM * gL > 0:
            lamL = lamM
            gL = valM
        else:
            lamR = lamM
            gR = valM
    
    lamM = 0.5*(lamL + lamR)
    return np.clip(beta + lamM*y, 0, C)

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
    return np.where(margin_vals - 1.0 <= 0)[0]

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

def make_toy_data(n=100, random_state=None, **kwargs):
    """
    Create a simple linearly separable dataset for SVM testing.
    
    Parameters:
    -----------
    n : int
        Total number of samples (approximately half per class)
    random_state : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    X : ndarray of shape (n, 2)
        Feature matrix with 2 dimensions for easy visualization
    y : ndarray of shape (n,)
        Class labels in {-1, 1}
    """
    # Create a simple separable dataset
    w = np.array([1.0, 1.0])  # normal vector for hyperplane
    b = 0.0                   # intercept
    n_A = n // 2             # number of samples for class +1
    n_B = n - n_A            # number of samples for class -1
    margin = 1.0              # margin size
    
    # Use the existing test_linear function
    return test_linear(w, b, n_A, n_B, margin, seed=random_state, **kwargs)

# Non-linear dataset example
def make_nonlinear_data(n=100, random_state=0):
    np.random.seed(random_state)
    X1 = np.random.randn(n, 2) * 0.5
    X2 = np.random.randn(n, 2) * 0.5 + np.array([2.0, 2.0])
    X3 = np.random.randn(n, 2) * 0.5 + np.array([-2.0, 2.0])
    X = np.vstack([X1, X2, X3])
    y = np.array([+1]*n + [-1]*n + [+1]*n)
    return X, y  # Add this return statement

def plot_decision_boundary_2D(model, X, y, title="", save_path=None):
    """
    Plot decision boundary and support vectors for any SVM model
    that follows the BaseSVM interface.
    
    Parameters:
    -----------
    model : BaseSVM
        Fitted SVM model
    X : array-like of shape (n_samples, 2)
        Input samples (must be 2D for visualization)
    y : array-like of shape (n_samples,)
        Target labels
    title : str, optional
        Plot title
    """
    fig, (ax, ax_learn) = plt.subplots(1, 2, figsize=(12, 5), 
                                        gridspec_kw={'width_ratios': [2, 1]})
    # Class scatter plot
    ax.scatter(X[y==+1,0], X[y==+1,1], label="+1 class", marker='o')
    ax.scatter(X[y==-1,0], X[y==-1,1], label="-1 class", marker='s')
    
    # Create mesh
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    XX, YY = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid_points = np.c_[XX.ravel(), YY.ravel()]
    
    # Predict using model's decision function
    Z = model.decision_function(grid_points)
    Z = Z.reshape(XX.shape)
    
    # Plot decision boundary (Z=0) and margins (Z=±1)
    ax.contour(XX, YY, Z, levels=[-1.0, 0.0, 1.0],
              colors=['r', 'k', 'r'], linestyles=['--', '-', '--'])
    
    # Fill regions
    ax.contourf(XX, YY, Z, levels=[-1e9, -1, 1, 1e9], 
                alpha=0.2, colors=['#FFCCCC', '#CCCCFF', '#CCFFCC'])
    
    # Highlight support vectors
    sv_indices = model.get_support_vectors()
    if len(sv_indices) > 0:
        ax.scatter(X[sv_indices, 0], X[sv_indices, 1], 
                   s=100, facecolors='none', edgecolors='k', label="Support Vectors")
    else:
        print("No support vectors found.")
    
    ax.set_title(title)
    ax.legend()
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.grid(True, alpha=0.3)
    
    # Learning curve
    ax_learn.plot(model.obj_history, label="Objective function")
    ax_learn.set_title("Learning curve")
    ax_learn.set_xlabel("Iterations")
    ax_learn.set_ylabel("Objective value")
    ax_learn.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax_learn.legend()
    ax_learn.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    
    plt.show()

def plot_convergence(history, save_path=None):
    """
    Creates convergence plots for optimization algorithms.
    
    Parameters:
    -----------
    history : dict
        Dictionary containing optimization history with keys:
        - 'objective': list of objective function values
        - 'grad_norm': list of gradient norm values (optional)
        - 'step_size': list of step sizes (optional)
        - 'alpha_diff': list of parameter changes between iterations (optional)
    save_path : str, optional
        Path to save the figure
    """
    n_plots = sum(key in history for key in ['objective', 'grad_norm', 'step_size', 'alpha_diff'])
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
    
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    iterations = np.arange(1, len(history.get('objective', [])) + 1) if 'objective' in history else None
    
    # Plot objective function values
    if 'objective' in history:
        ax = axes[plot_idx]
        obj_values = history['objective']
        
        # Linear scale
        ax.plot(iterations, obj_values, 'b-', label='Objective')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Objective Value')
        ax.set_title('Objective Function Convergence')
        ax.grid(True, alpha=0.3)
        
        # Add log scale inset
        if len(obj_values) > 10:
            opt_val = min(obj_values) if min(obj_values) < 0 else 0
            inset = ax.inset_axes([0.55, 0.55, 0.4, 0.4])
            inset.semilogy(iterations, [abs(v - opt_val) + 1e-10 for v in obj_values], 'r-')
            inset.set_title('Log Scale', fontsize=8)
            inset.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # Plot gradient norm
    if 'grad_norm' in history:
        ax = axes[plot_idx]
        grad_values = history['grad_norm']
        
        ax.semilogy(iterations, grad_values, 'g-', label='Gradient Norm')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norm Convergence')
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # Plot step size
    if 'step_size' in history:
        ax = axes[plot_idx]
        step_sizes = history['step_size']
        
        ax.semilogy(iterations[:len(step_sizes)], step_sizes, 'm-', label='Step Size')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Step Size')
        ax.set_title('Step Size Variation')
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # Plot alpha differences
    if 'alpha_diff' in history:
        ax = axes[plot_idx]
        alpha_diffs = history['alpha_diff']
        
        ax.semilogy(iterations[:len(alpha_diffs)], alpha_diffs, 'c-', label='Parameter Change')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('||α_{k+1} - α_k||')
        ax.set_title('Parameter Change')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    
    return fig, axes

def analyze_convergence(history, window_size=10, save_path=None):
    """
    Analyzes convergence rates and provides diagnostics for optimization algorithms.
    
    Parameters:
    -----------
    history : dict
        Dictionary containing optimization history with keys:
        - 'objective': list of objective function values
        - 'grad_norm': list of gradient norm values (optional)
        - 'step_size': list of step sizes (optional)
        - 'alpha_diff': list of parameter changes between iterations (optional)
    window_size : int, optional
        Window size for rolling rate calculations
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    dict
        Dictionary containing convergence analysis metrics
    """
    results = {}
    
    if 'objective' not in history:
        print("Objective function history not provided, cannot analyze convergence")
        return results
    
    obj_values = np.array(history['objective'])
    iterations = np.arange(1, len(obj_values) + 1)
    
    # Compute optimal value estimate (minimum observed)
    opt_val_est = min(obj_values)
    results['optimal_value_est'] = opt_val_est
    
    # Calculate distance to optimum
    dist_to_opt = np.abs(obj_values - opt_val_est) + 1e-15  # Add small constant to avoid log(0)
    results['dist_to_opt'] = dist_to_opt
    
    # Compute convergence rates in sliding windows
    lin_rates = []
    quad_rates = []
    for i in range(window_size, len(obj_values)):
        window = dist_to_opt[i-window_size:i]
        # Linear rate: f_{k+1} - f* ≤ r * (f_k - f*)
        if all(window[:-1] > 0):
            rates = window[1:] / window[:-1]
            lin_rates.append(np.mean(rates))
        
        # Quadratic rate: f_{k+1} - f* ≤ c * (f_k - f*)^2
        if all(window[:-1] > 0):
            rates = window[1:] / (window[:-1]**2)
            quad_rates.append(np.mean(rates))
    
    results['linear_rates'] = lin_rates
    results['quadratic_rates'] = quad_rates
    
    # Estimate overall convergence rate
    if len(lin_rates) > 0:
        results['avg_linear_rate'] = np.mean(lin_rates)
    if len(quad_rates) > 0:
        results['avg_quadratic_rate'] = np.mean(quad_rates)
    
    # Create convergence plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Objective value vs. iterations
    ax = axes[0, 0]
    ax.plot(iterations, obj_values, 'b-')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Objective Value')
    ax.set_title('Objective Function Convergence')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Log of distance to optimum
    ax = axes[0, 1]
    ax.semilogy(iterations, dist_to_opt, 'r-')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('log|f(x_k) - f*|')
    ax.set_title('Distance to Optimum (Log Scale)')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Estimated local convergence rate
    ax = axes[1, 0]
    if len(lin_rates) > 0:
        ax.plot(range(window_size, window_size + len(lin_rates)), lin_rates, 'g-', label='Linear Rate')
        ax.axhline(results.get('avg_linear_rate', 0), color='g', linestyle='--', 
                  label=f'Avg: {results.get("avg_linear_rate", 0):.4f}')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Rate')
        ax.set_title(f'Local Linear Convergence Rate (window={window_size})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Additional convergence diagnostics
    ax = axes[1, 1]
    
    # Include gradient norm if available
    if 'grad_norm' in history:
        grad_norms = history['grad_norm']
        ax.semilogy(iterations[:len(grad_norms)], grad_norms, 'c-', label='Gradient Norm')
    
    # Include parameter changes if available
    if 'alpha_diff' in history:
        alpha_diffs = history['alpha_diff']
        ax.semilogy(iterations[:len(alpha_diffs)], alpha_diffs, 'm-', label='||α_{k+1} - α_k||')
    
    ax.set_xlabel('Iterations')
    ax.set_title('Additional Convergence Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    
    results['figure'] = fig
    
    return results

