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
