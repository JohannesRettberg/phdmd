import cvxpy as cp
import numpy as np

def PHformLMIs(sys, display=1, standard=0):
    n = sys['A'].shape[0]
    m = sys['B'].shape[1]
    
    A = sys['A']
    B = sys['B']
    E = sys.get('E', np.eye(n))
    C = sys['C']
    D = sys['D']
    
    # Define variables
    X = cp.Variable((n, n), symmetric=True)
    delta = cp.Variable()
    
    # Define the constraints
    constraints = [
        cp.bmat([
            [-A.T @ X - X.T @ A, -X.T @ B + C.T],
            [-B.T @ X + C, D + D.T]
        ]) + delta * np.eye(n + m) >> 0,
        
        E.T @ X + delta * np.eye(n) >> 0
    ]
    
    # Define the objective function
    objective = cp.Minimize(cp.norm(delta))
    
    # Form and solve the problem
    problem = cp.Problem(objective, constraints)
    
    if display == 1:
        problem.solve(verbose=True)
    else:
        problem.solve()
    
    if display == 1:
        if np.linalg.cond(X.value) <= 1e9 and np.linalg.norm(delta.value) < 1e-6:
            print('The system admits a DH-form.')
        else:
            print('The system does not admit a DH-form, an approximation was provided.')
    
    return X.value, delta.value