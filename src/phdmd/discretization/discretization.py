import numpy as np
import scipy
import time

from tqdm import tqdm

from pymor.algorithms.to_matrix import to_matrix
from pymor.models.iosys import LTIModel, PHLTIModel


def discretize(lti, U, T, x0, method='implicit_midpoint', return_dXdt=False):
    """
    Discretize a continuous-time linear time-invariant system.

    Parameters
    ----------
    lti : pymor.models.iosys.PHLTIModel
        The LTI system to discretize.
    U : np.ndarray or callable
        The input signal. If callable, it must take a single argument T and
        return a 2D array of shape (dim_input, len(T)).
    T : np.ndarray
        The time instants at which to compute the data.
    x0 : np.ndarray
        The initial state.
    method : str
        The method to use for the discretization. Available methods are:
        'implicit_midpoint', 'explicit_euler', 'explicit_midpoint', 'scipy_solve_ivp'.
    return_dXdt : bool
        Whether to return the time derivative of the state.

    Returns
    -------
    U : np.ndarray
        The input signal.
    X : np.ndarray
        The state data.
    Y : np.ndarray
        The output data.
    """
    assert isinstance(lti, PHLTIModel) or isinstance(lti, LTIModel)
    if isinstance(lti, PHLTIModel):
        lti = lti.to_lti()

    if isinstance(U,list):
        U_is_list = True
        n_scenarios = len(U)
        U_temp = U.copy()[0] 
        U_list = U.copy()                
    else:
        U_is_list = False
        U_temp = U
        n_scenarios = 1

    if not isinstance(U_temp, np.ndarray):
        n_u = U_temp(T).shape[0]
        if U_temp(T).ndim < 2:
            n_u = 1
    else:
        n_u = U_temp.shape[0]

    if return_dXdt:
        X = np.zeros((lti.order, len(T),n_scenarios))
        U = np.zeros((n_u, len(T), n_scenarios))
        Y = np.zeros((n_u, len(T), n_scenarios))
        dXdt = np.zeros((lti.order, len(T),n_scenarios))
    else:
        X = np.zeros((lti.order, len(T),n_scenarios))
        U = np.zeros((n_u, len(T), n_scenarios))
        Y = np.zeros((n_u, len(T), n_scenarios))
    for i_scenario in range(n_scenarios):
        if U_is_list:
            U_temp = U_list[i_scenario]
        else:
            pass # keep U_temp
        match method:
            case 'implicit_midpoint':
                if return_dXdt:
                    U[:,:,i_scenario], X[:,:,i_scenario], Y[:,:,i_scenario], dXdt[:,:,i_scenario] =  implicit_midpoint(lti, U_temp, T, x0, return_dXdt)
                else:
                    U[:,:,i_scenario], X[:,:,i_scenario], Y[:,:,i_scenario] =  implicit_midpoint(lti, U_temp, T, x0, return_dXdt)
            case 'explicit_euler':
                if return_dXdt:
                    U[:,:,i_scenario], X[:,:,i_scenario], Y[:,:,i_scenario], dXdt[:,:,i_scenario] =  explicit_euler(lti, U_temp, T, x0, return_dXdt)
                else:
                    U[:,:,i_scenario], X[:,:,i_scenario], Y[:,:,i_scenario] =  explicit_euler(lti, U_temp, T, x0, return_dXdt)
            case 'explicit_midpoint':
                if return_dXdt:
                    U[:,:,i_scenario], X[:,:,i_scenario], Y[:,:,i_scenario], dXdt[:,:,i_scenario] =  explicit_midpoint(lti, U_temp, T, x0, return_dXdt)
                else:
                    U[:,:,i_scenario], X[:,:,i_scenario], Y[:,:,i_scenario] =  explicit_midpoint(lti, U_temp, T, x0, return_dXdt)
            case _:
                if return_dXdt:
                    U[:,:,i_scenario], X[:,:,i_scenario], Y[:,:,i_scenario], dXdt[:,:,i_scenario] =  scipy_solve_ivp(lti, U_temp, T, x0, method, return_dXdt)
                else:
                    U[:,:,i_scenario], X[:,:,i_scenario], Y[:,:,i_scenario] =  scipy_solve_ivp(lti, U_temp, T, x0, method, return_dXdt)
    if return_dXdt:
        return np.reshape(np.transpose(U,(0,2,1)),(n_u,len(T)*n_scenarios)),np.reshape(np.transpose(X,(0,2,1)),(lti.order,len(T)*n_scenarios)), np.reshape(np.transpose(Y,(0,2,1)),(n_u,len(T)*n_scenarios)), np.reshape(np.transpose(dXdt,(0,2,1)),(lti.order,len(T)*n_scenarios))
    else:
        return np.reshape(np.transpose(U,(0,2,1)),(n_u,len(T)*n_scenarios)),np.reshape(np.transpose(X,(0,2,1)),(lti.order,len(T)*n_scenarios)), np.reshape(np.transpose(Y,(0,2,1)),(n_u,len(T)*n_scenarios))

def implicit_midpoint(lti, U, T, x0, return_dXdt=False):
    """
    Discretize a continuous-time linear time-invariant system using the implicit midpoint method.

    Parameters
    ----------
    lti : pymor.models.iosys.LTIModel
        The LTI system to discretize.
    U : np.ndarray or callable
        The input signal. If callable, it must take a single argument T and
        return a 2D array of shape (dim_input, len(T)).
    T : np.ndarray
        The time instants at which to compute the data.
    x0 : np.ndarray
        The initial state.
    return_dXdt : bool
        Whether to return the time derivative of the state.

    Returns
    -------
    U : np.ndarray
        The input signal.
    X : np.ndarray
        The state data.
    Y : np.ndarray
        The output data.
    """
    if not isinstance(U, np.ndarray):
        U = U(T)
        if U.ndim < 2:
            U = U[np.newaxis, :]

    delta = T[1] - T[0]

    M = to_matrix(lti.E - delta / 2 * lti.A)
    AA = to_matrix(lti.E + delta / 2 * lti.A)
    E = to_matrix(lti.E, format='dense')
    A = to_matrix(lti.A)
    B = to_matrix(lti.B)
    C = to_matrix(lti.C)
    D = to_matrix(lti.D, format='dense')

    X = np.zeros((lti.order, len(T)))
    X[:, 0] = x0.ravel()

    
    M_issparse = scipy.sparse.issparse(M) 
    # LU decomposition  
    if M_issparse:
        if M.getformat() == 'csr':
            # convert to csc to prevent splu giving a warning 
            M = scipy.sparse.csc_matrix(M)
        invM_sparse = scipy.sparse.linalg.splu(M)
    else:
        lu_dense, piv_dense = scipy.linalg.lu_factor(M)
        # AA as matrix lead to errors in matrix multiplication AA of shape (n,k) and X[:,i] of shape (k,) lead to shape (1,n) 
        AA = np.array(AA)
    
    for i in tqdm(range(len(T) - 1)):
        U_midpoint = 1 / 2 * (U[:, i] + U[:, i + 1])
        if M_issparse:
            X[:, i + 1] = invM_sparse.solve(AA @ X[:, i] + delta * B @ U_midpoint)
        else:
            X[:, i + 1] = scipy.linalg.lu_solve((lu_dense, piv_dense), AA @ X[:, i] + delta * B @ U_midpoint)

    Y = C @ X + D @ U

    if not return_dXdt:
        return U, X, Y
    else:
        dXdt = np.linalg.solve(E, A @ X + B @ U)
        return U, X, Y, dXdt

    # import time
    # import scipy
    # M_dense = M.todense()
    # M_array = M.toarray()
    # lu_dense, piv_dense = scipy.linalg.lu_factor(M_array)
    # invM_sparse = scipy.sparse.linalg.splu(M)
    # start = time.time()
    # X[:, i + 1] = scipy.sparse.linalg.spsolve(M, AA @ X[:, i] + delta * B @ U_midpoint)
    # end = time.time()
    # print(f'Scipy sparse solve approach takes {end - start} s and result is {X[:, i + 1]}')
    # start = time.time()
    # X[:, i + 1] = scipy.linalg.solve(M_dense, AA @ X[:, i] + delta * B @ U_midpoint)
    # end = time.time()
    # print(f'Scipy linalg approach with dense matrix takes {end - start} s and result is {X[:, i + 1]}')
    # start = time.time()
    # X[:, i + 1] = scipy.linalg.solve(M_array, AA @ X[:, i] + delta * B @ U_midpoint)
    # end = time.time()
    # print(f'Scipy linalg approach with np array takes {end - start} s and result is {X[:, i + 1]}')
    # # start = time.time()
    # # X[:, i + 1] = np.linalg.solve(M, AA @ X[:, i] + delta * B @ U_midpoint)
    # # end = time.time()
    # # print(end - start)
    # start = time.time()
    # X[:, i + 1] = np.linalg.solve(M_dense, AA @ X[:, i] + delta * B @ U_midpoint)
    # end = time.time()
    # print(f'Numpy linalg approach with dense matrix takes {end - start} s and result is {X[:, i + 1]}')
    # start = time.time()
    # X[:, i + 1] = np.linalg.solve(M_array, AA @ X[:, i] + delta * B @ U_midpoint)
    # end = time.time()
    # print(f'Numpy linalg approach with np array takes {end - start} s and result is {X[:, i + 1]}') 
    # start = time.time()
    # X[:, i + 1] = scipy.linalg.lu_solve((lu_dense, piv_dense), AA @ X[:, i] + delta * B @ U_midpoint)
    # end = time.time()
    # print(f'LU solve approach with np array takes {end - start} s and result is {X[:, i + 1]}') 
    # start = time.time()
    # X[:, i + 1] = invM_sparse.solve(AA @ X[:, i] + delta * B @ U_midpoint)
    # end = time.time()
    # print(f'LU solve approach with sparse matrix takes {end - start} s and result is {X[:, i + 1]}') 
    

def explicit_euler(lti, U, T, x0, return_dXdt=False):
    """
    Discretize a continuous-time linear time-invariant system using the explicit Euler method.

    Parameters
    ----------
    lti : pymor.models.iosys.LTIModel
        The LTI system to discretize.
    U : np.ndarray or callable
        The input signal. If callable, it must take a single argument T and
        return a 2D array of shape (dim_input, len(T)).
    T : np.ndarray
        The time instants at which to compute the data.
    x0 : np.ndarray
        The initial state.
    return_dXdt : bool
        Whether to return the time derivative of the state.

    Returns
    -------
    U : np.ndarray
        The input signal.
    X : np.ndarray
        The state data.
    Y : np.ndarray
        The output data.
    """
    if not isinstance(U, np.ndarray):
        U = U(T)
        if U.ndim < 2:
            U = U[np.newaxis, :]

    delta = T[1] - T[0]

    E = to_matrix(lti.E, format='dense')
    A = to_matrix(lti.A)
    B = to_matrix(lti.B)
    C = to_matrix(lti.C)
    D = to_matrix(lti.D, format='dense')

    X = np.zeros((lti.order, len(T)))
    X[:, 0] = x0

    for i in tqdm(range(len(T) - 1)):
        X[:, i + 1] = X[:, i] + delta * np.linalg.solve(E, A @ X[:, i] + B @ U[:, i])

    Y = C @ X + D @ U

    if not return_dXdt:
        return U, X, Y
    else:
        dXdt = np.linalg.solve(E, A @ X + B @ U)
        return U, X, Y, dXdt


def explicit_midpoint(lti, U, T, x0, return_dXdt=False):
    """
    Discretize a continuous-time linear time-invariant system using the explicit midpoint method.

    Parameters
    ----------
    lti : pymor.models.iosys.LTIModel
        The LTI system to discretize.
    U : np.ndarray or callable
        The input signal. If callable, it must take a single argument T and
        return a 2D array of shape (dim_input, len(T)).
    T : np.ndarray
        The time instants at which to compute the data.
    x0 : np.ndarray
        The initial state.
    return_dXdt : bool
        Whether to return the time derivative of the state.

    Returns
    -------
    U : np.ndarray
        The input signal.
    X : np.ndarray
        The state data.
    Y : np.ndarray
        The output data.
    """
    if not isinstance(U, np.ndarray):
        U = U(T)
        if U.ndim < 2:
            U = U[np.newaxis, :]

    delta = T[1] - T[0]

    E = to_matrix(lti.E, format='dense')
    A = to_matrix(lti.A)
    B = to_matrix(lti.B)
    C = to_matrix(lti.C)
    D = to_matrix(lti.D, format='dense')

    X = np.zeros((lti.order, len(T)))
    X[:, 0] = x0

    for i in tqdm(range(len(T) - 1)):
        X_ = X[:, i] + delta * np.linalg.solve(E, A @ X[:, i] + B @ U[:, i])
        X[:, i + 1] = X[:, i] + delta * np.linalg.solve(E, A @ X_ + B @ (1 / 2 * (U[:, i] + U[:, i + 1])))

    Y = C @ X + D @ U

    if not return_dXdt:
        return U, X, Y
    else:
        dXdt = np.linalg.solve(E, A @ X + B @ U)
        return U, X, Y, dXdt


def scipy_solve_ivp(lti, u, T, x0, method='RK45', return_dXdt=False):
    E = to_matrix(lti.E, format='dense')
    A = to_matrix(lti.A)
    B = to_matrix(lti.B)
    C = to_matrix(lti.C)
    D = to_matrix(lti.D, format='dense')

    U = u(T)
    if U.ndim < 2:
        U = U[np.newaxis, :]

    from scipy.integrate import solve_ivp

    def f(t, x, u):
        return np.linalg.solve(E, A @ x + B @ u(t))

    sol = solve_ivp(f, (T[0], T[-1]), x0, t_eval=T, method=method, args=(u,))
    X = sol.y
    Y = C @ X + D @ U

    if not return_dXdt:
        return U, X, Y
    else:
        dXdt = np.linalg.solve(E, A @ X + B @ U)
        return U, X, Y, dXdt
