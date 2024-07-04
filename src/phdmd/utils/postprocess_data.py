import logging
import numpy as np

from pymor.algorithms.to_matrix import to_matrix


# print marices of original and identified system
def compare_matrices(exp, lti_id_dict, zero_threshold=1e-10, latex_output=False):
    # set printoptions
    np.set_printoptions(formatter={"float": "\t{: 0.2e}\t".format})

    # print H
    logging.info(f"%%%%%%%%%%%%%%%%% Compare H matrix %%%%%%%%%%%%%%%%%%%%%%%")
    E_orig = to_matrix(exp.fom.E).copy()
    E_orig[np.abs(E_orig) < zero_threshold] = 0
    E_id = to_matrix(lti_id_dict.E).copy()
    E_id[np.abs(E_id) < zero_threshold] = 0
    print(E_orig)
    print(E_id)

    if latex_output:
        print(bmatrix(E_orig))
        print(bmatrix(E_id))

    # print Q
    logging.info(f"%%%%%%%%%%%%%%%%% Compare Q matrix %%%%%%%%%%%%%%%%%%%%%%%")
    Q_orig = to_matrix(exp.fom.Q).copy()
    Q_orig[np.abs(Q_orig) < zero_threshold] = 0
    Q_id = to_matrix(lti_id_dict.Q).copy()
    Q_id[np.abs(Q_id) < zero_threshold] = 0
    print(Q_orig)
    print(Q_id)
    if latex_output:
        print(bmatrix(Q_orig))
        print(bmatrix(Q_id))

    # print J
    logging.info(f"%%%%%%%%%%%%%%%%% Compare J matrix %%%%%%%%%%%%%%%%%%%%%%%")
    J_orig = to_matrix(exp.fom.J).copy()
    J_orig[np.abs(J_orig) < zero_threshold] = 0
    J_id = to_matrix(lti_id_dict.J).copy()
    J_id[np.abs(J_id) < zero_threshold] = 0
    print(J_orig)
    print(J_id)
    if latex_output:
        print(bmatrix(J_orig))
        print(bmatrix(J_id))

    # print R
    logging.info(f"%%%%%%%%%%%%%%%%% Compare R matrix %%%%%%%%%%%%%%%%%%%%%%%")
    R_orig = to_matrix(exp.fom.R).copy()
    R_orig[np.abs(R_orig) < zero_threshold] = 0
    R_id = to_matrix(lti_id_dict.R).copy()
    R_id[np.abs(R_id) < zero_threshold] = 0
    print(R_orig)
    print(R_id)
    if latex_output:
        print(bmatrix(R_orig))
        print(bmatrix(R_id))

    # print G
    logging.info(f"%%%%%%%%%%%%%%%%% Compare G matrix %%%%%%%%%%%%%%%%%%%%%%%")
    G_orig = to_matrix(exp.fom.G).copy()
    G_orig[np.abs(G_orig) < zero_threshold] = 0
    G_id = to_matrix(lti_id_dict.G).copy()
    G_id[np.abs(G_id) < zero_threshold] = 0
    print(G_orig)
    print(G_id)
    if latex_output:
        print(bmatrix(G_orig))
        print(bmatrix(G_id))

    # print P
    logging.info(f"%%%%%%%%%%%%%%%%% Compare P matrix %%%%%%%%%%%%%%%%%%%%%%%")
    P_orig = to_matrix(exp.fom.P).copy()
    P_orig[np.abs(P_orig) < zero_threshold] = 0
    P_id = to_matrix(lti_id_dict.P).copy()
    P_id[np.abs(P_id) < zero_threshold] = 0
    print(P_orig)
    print(P_id)

    # print S
    logging.info(f"%%%%%%%%%%%%%%%%% Compare S matrix %%%%%%%%%%%%%%%%%%%%%%%")
    S_orig = to_matrix(exp.fom.S).copy()
    S_orig[np.abs(S_orig) < zero_threshold] = 0
    S_id = to_matrix(lti_id_dict.S).copy()
    S_id[np.abs(S_id) < zero_threshold] = 0
    print(S_orig)
    print(S_id)

    # print N
    logging.info(f"%%%%%%%%%%%%%%%%% Compare N matrix %%%%%%%%%%%%%%%%%%%%%%%")
    N_orig = to_matrix(exp.fom.N).copy()
    N_orig[np.abs(N_orig) < zero_threshold] = 0
    N_id = to_matrix(lti_id_dict.N).copy()
    N_id[np.abs(N_id) < zero_threshold] = 0
    print(N_orig)
    print(N_id)


def bmatrix(a):
    """Returns a LaTeX bmatrix
    Taken from https://stackoverflow.com/questions/17129290/numpy-2d-and-1d-array-to-latex-bmatrix
    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError("bmatrix can at most display two dimensions")
    lines = str(a).replace("[", "").replace("]", "").splitlines()
    rv = [r"\begin{bmatrix}"]
    rv += ["  " + " & ".join(l.split()) + r"\\" for l in lines]
    rv += [r"\end{bmatrix}"]
    return "\n".join(rv)
