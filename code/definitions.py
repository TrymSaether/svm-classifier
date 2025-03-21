import numpy as np

y = np.ones(10)
beta = np.full_like(y, 0.5)
C = 1
lambda_minus = 0
lambda_plus = C
delta = 0.01
epsilon = 0.01

alpha = lambda y, lamb: np.minimum(np.maximum(0,beta + lamb*y), C)

def find_lambda(y, lambda_minus, lambda_plus, delta, epsilon):
    while True:
        alpha_minus = alpha(y,lambda_minus)
        alpha_plus = alpha(y,lambda_plus)
        
        dot_minus = np.dot(y, alpha_minus)
        dot_plus = np.dot(y, alpha_plus)
        
        if dot_minus > 0:
            lambda_plus = lambda_minus
            lambda_minus -= delta
        elif dot_plus < 0:
            lambda_minus = lambda_plus
            lambda_plus += delta
        else:
            break
    
    while True:
        lambda_hat = 0.5 * (lambda_minus + lambda_plus)
        alpha_hat = alpha(y,lambda_hat)
        
        if abs(np.dot(y, alpha_hat)) < epsilon:
            return lambda_hat
        elif np.dot(y, alpha_hat) < 0:
            lambda_minus = lambda_hat
        else:
            lambda_plus = lambda_hat

lambda_star = find_lambda(y, lambda_minus, lambda_plus, delta, epsilon)
alpha_star = alpha(y, lambda_star)


