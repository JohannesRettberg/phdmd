import numpy as np
from scipy import linalg
import logging


# %% define functions
def create_mass_spring_damper_system(
    n_mass,
    mass_vals=1,
    damp_vals=1,
    stiff_vals=1,
    input_vals=None,
    system="ph",
    use_Berlin=True,
    use_cholesky_like_fact=False,
):
    """
    Creates a mass-spring-damper system
    :param n_mass: number of masses, i.e. second order system size (integer value)
    :param mass_vals: mass values either (n_mass,) array or scalar value (same value applied to all masses)
    :param damp_vals: damping values either (n_mass,) array or scalar value (same value applied to all dampers)
    :param stiff_vals: stiffness values either (n_mass,) array or scalar value (same value applied to all springs)
    :param input_vals: array of size (number of inputs,) creates an array of size (n_mass,len(input_vals)) with ones at indices from input_vals (index of excited mass)
    :param system: system output matrices, string with either {'ph'}, '2nd', or 'ss' for port-Hamiltonian, second order or state-space matrices
    :return: return the matrices that were requested from 'system'
    """
    # number of states
    n = 2 * n_mass
    # create mass matrix
    if isinstance(mass_vals, (list, tuple, np.ndarray)):
        M = np.diag(mass_vals)
    else:  # scalar value
        M = np.eye(n_mass) * mass_vals

    # create damping matrix
    if isinstance(mass_vals, (list, tuple, np.ndarray)):
        D = np.diag(damp_vals)
    else:  # scalar value
        D = np.eye(n_mass) * damp_vals

    # create stiffness matrix
    if not isinstance(stiff_vals, (list, tuple, np.ndarray)):
        # scalar value
        stiff_vals = np.ones(n_mass) * stiff_vals
    K = np.zeros((n_mass, n_mass))
    K[:, :] = np.diag(stiff_vals[:])
    K[1:, 1:] += np.diag(stiff_vals[:-1])
    K += -np.diag(stiff_vals[:-1], -1)
    K += -np.diag(stiff_vals[:-1], 1)

    # create input vector
    if input_vals is not None:
        if isinstance(input_vals, int):
            # single input
            assert input_vals <= n_mass
            B_2nd = np.zeros((n_mass, 1))
            B_2nd[input_vals, 0] = 1
        else:
            assert max(list(input_vals)) <= n_mass
            B_2nd = np.zeros((n_mass, len(input_vals)))
            B_2nd[input_vals, np.arange(len(input_vals))] = 1
    else:
        B_2nd = np.zeros((n_mass))

    # number of inputs
    n_u = B_2nd.shape[-1]

    # create state-space system
    M_inv = np.linalg.inv(M)
    A = np.block(
        [[np.zeros((n_mass, n_mass)), np.eye(n_mass)], [-M_inv @ K, -M_inv @ D]]
    )

    # scale input with M_inv
    B = np.concatenate((np.zeros(B_2nd.shape), M_inv @ B_2nd), axis=0)

    # convert to port-Hamiltonian system
    J = np.diag(np.ones(n_mass), n_mass)
    J += -np.diag(np.ones(n_mass), -n_mass)

    R = linalg.block_diag(np.zeros((n_mass, n_mass)), D)

    Q = linalg.block_diag(K, M_inv)

    # B_ph cancels out M with M_inv due to momentum description
    B_ph = np.concatenate((np.zeros(B_2nd.shape), B_2nd), axis=0)

    H = np.eye(n)
    G = B_ph
    P = np.zeros_like(G)
    S = np.zeros((n_u, n_u))
    N = np.zeros_like(S)

    if use_Berlin:
        # bring Q to left-hand side
        H = Q.T @ H
        J = Q.T @ J @ Q
        R = Q.T @ R @ Q
        G = Q.T @ G
        P = Q.T @ P
        Q = np.eye(J.shape[0])

    if use_cholesky_like_fact:
        if use_Berlin:
            # assume skew-symmetric matrix J of form [[0, E.T],[-E 0]]
            J12 = J[:n_mass, n_mass : 2 * n_mass].T
            # transformation matrix
            T = np.block(
                [
                    [np.zeros((n_mass, n_mass)), J12.T],
                    [-np.eye(n_mass), np.zeros((n_mass, n_mass))],
                ]
            )

            T_inv = np.linalg.inv(T)

            J = T_inv.T @ J @ T_inv
            # check transformation (should be in Poisson form)
            if np.allclose(
                J,
                np.block(
                    [
                        [np.zeros((n_mass, n_mass)), np.eye(n_mass)],
                        [-np.eye(n_mass), np.zeros((n_mass, n_mass))],
                    ]
                ),
            ):
                print(f"J successfully transformed to Poisson form.")
            else:
                print(f"J could not be transformed to Poisson form.")

            H = T_inv.T @ H @ T_inv
            R = T_inv.T @ R @ T_inv
            G = T_inv.T @ G
            P = T_inv.T @ P
        else:
            # J already in desired format for Q energy
            pass

    if system == "ph":
        return H, J, R, G, P, S, N, Q
    elif system == "2nd":
        return M, D, K, B_2nd
    elif system == "ss":
        return A, B, M
    else:
        raise Exception("system input not known. Choose either 'ph','2nd' or 'ss' ")
