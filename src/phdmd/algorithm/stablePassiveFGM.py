import numpy as np
from scipy.linalg import eig, norm
from time import time, process_time
import logging
import cvxpy as cp

from pymor.models.iosys import LTIModel, PHLTIModel


# % function [PHform,e,t] = stablePassiveFGM(sys,options);
# %
# % Find the nearest port-Hamiltonian system to a given system (E,A,B,C,D)
# % solving the following optimization problem
# %
# % min_{ J,R,Q,M,F,P,S,N }   ||A - (J-R)Q||_F^2    + ||B - (T-P)||_F^2
# %                         + ||C - (F+P)^T Q||_F^2 + ||D - (S+N)||_F^2
# %                         + ||M-E||_F^2 ,
# % where J = -J', N=-N', Q invertible, Q^T M and K = [R P; P^T S] PSD
# %
# % For the standard form, we impose M = I_n = E.
# %
# % For the general form, we reformulate the problem using Z = M^TQ PSD:
# %
# % min_{ J,R,Q,Z,F,P,S } ||A - (J-R)Q||_F^2 + ||E^T - ZQ^-1||_F^2
# %       + ||B - (F-P)||_F^2 + ||C - (F+P)^T Q||_F^2 + ||(D+D')/2-S||_F^2
# % where J = -J', Q invertible, Z and [R P; P^T S] PSD
# %
# % It uses a *fast gradient method* (FGM) from smooth convex optimization,
# % with a safety restarting procedure (the problem is not convex).
# %
# % See the paper 'Finding the nearest positive-real system',
# % by Nicolas Gillis and Punit Sharma, 2017.
# %
# % ****** Input ******
# % sys.A,sys.B,sys.C,sys.D,sys.E :  a dynamical system
# %                                  If sys.E is not specified, the system is
# %                                  assumed to be standard with sys.E = I_n
# %
# % ---Options---
# % standard   : if standard == 1, M = I_n is fixed (E is not perturbed)
# %            -default = 0.
# % maxiter    : the maximum number of iterations performed by the algorithm
# %            -default = Inf.
# % timemax    : the maximum time in seconds alloted to the algorithm
# %            -default = 10.
# % posdef     : lower bound (>= 0) for the smallest eigenvalue of K and Z.
# %              For posdef>0, it will make K and Z positive definite
# %              avoiding Q and K to converge to a rank-deficient matrix
# %              hence will generate an admissible ESPR system.
# %            -default = 0.
# % PHform     : this is used as an initial starting point where PHform
# %              contains either only PHform.Q and the other matrices are
# %              initialized solving an optimization problem, or
# %              PHform.{J,R,Q,S,Z,P,S,F}.
# %              Otherwise options.init is used
# % init       : initial matrices used for the algorithm
# %            -default: =1, standard init.
# %                      Q = I_n, (identity matrix)
# %                      P = 0, (zero matrix)
# %                      (J,R,S,P,Z) are optimal w.r.t. to Q and P.
# %            -init=2,3: solve the LMIs (cf. PHformLMIs.m) and set
# %                      Q = X = solution of the LMIs,
# %                      Z = ProjectPSD(E'*Q), (for general systems)
# %               *init=2: (J,R,F,S,P) matrices chosen w.r.t. system in PH-form
# %                      with X as the solution of the LMIs (+projection);
# %                      see PHformJRPSF.m.
# %               *init=3: compute the optimal (J,R,F,S,P); see solveJRFPS.m
# %              If the user provides the matrices (J,Q,R,S,F,P) in
# %                     options.PHform.J, options.PHform.Q, etc, then these initial
# %                     matrices are used.
# %                     If only options.Q is provided, then the other
# %                     matrices are computed as in init=3.
# % alpha0 in (0,1) : parameter of the fast gradient method
# %                 - default = 0.5
# % lsparam and lsitermax: line-search parameters
# %             The line-search starts with a step estimated using a lower
# %             bound of the Lipschit constant. In most our experiments, this
# %             step size is often acceptable (it allows decrease). Otherwise,
# %             * lsparam is the factor of reduction of the step size until
# %               the objective function decreases: step <- step/lsparam
# %             -default = 1.5
# %             * lsitermax is the maximum number of times the step size is
# %               decreased (except for the first iteration)
# %             -default = 20
# % gradient in {0,1} : =1 -> a standard gradient scheme is used
# %                     otherwise the fast gradient method (FGM) is used.
# %             -default = 0
# % weight in R^5_+ : changes the objective function to give more importance
# %                   to the different terms in the objective function:
# %                 weight(1)*||A-(J-R)Q||_F^2
# %               + weight(2)*||B-(T-P)||_F^2
# %               + weight(3)*||C-(F+P)^T Q||_F^2
# %               + weight(4)*||(D+D')/2-S||_F^2
# %               (+ weight(5)*||E^T-Z*Q^-1||_F^2 for non-standard systems)
# %             -default = weight = ones(1,5);
# % display in {0,1} : =1 displays the evolution of the objective, and
# %                       SDPT3 log, 0 otherwise
# %             -default = 1
# %
# % ****** Output ******
# % PHform.(J,R,Q,F,P,S,N,M,Z) such that (Ep,Ap,Bp,Cp,Dp) is close to the
# % original syste (E,A,B,C,D) and is passive, where
# % Ap = (J-R)*Q,
# % Ep =  I_n if standard == 1, otherwise Ep = M = (ZQ^-1)^T,
# % Bp = (F-P),
# % Cp = (F+P)'*Q,
# % Dp = (S+N).
# %
# % e          : evolution of the objective function
# % t          : cputime at each iteration
# %            --plot(t,e) displays the error over time
# %            --plot(e) displays the error over iterations


def stablePassiveFGM(sys, options):
    # % PHform contains the matrices (J,R,Q,F,P,S,N,M,Z) defining the PH form
    # %
    # % sys contains A,E,B,C,D. If E is not specified, it is assumed it is a
    # % standard system with E = I_n

    cput = process_time()
    n = sys["A"].shape[0]
    m = sys["B"].shape[1]

    # % Simplify notation
    A = sys["A"]
    if sys["B"] is None:  # No input to the system
        B = np.zeros((n,))
    else:
        B = sys["B"]

    if sys["E"] is None:
        E = np.eye(n)
    else:
        E = sys["E"]

    if sys["C"] is None:
        C = np.zeros((n,))
    else:
        C = sys["C"]

    D = sys["D"]

    # Optimal N is given by
    N = (D - D.T) / 2

    options = set_default_options(options)

    if options["weight"].min() < 0:
        raise ValueError("The parameter options.weight must be nonnegative.")

    if "Q" in options["PHform"]:
        if options["standard"] == 1 and np.min(np.linalg.eigvals(Q)) < 0:
            logging.warning("For standard systems, Q must be PSD. It was projected.")
            options["PHform"]["Q"] = projectPSD(options["PHform"]["Q"])

        # % Case 1: all input matrices are provided
        # %         This is particularly useful if the user wants to reinitialize
        # %         the algorithm with the solution of the previous run.
        if all(key in options["PHform"] for key in ["J", "R", "P", "S", "F"]):
            PHform = options["PHform"]
            J = PHform["J"]
            R = PHform["R"]
            Q = PHform["Q"]
            if options["standard"] != 1:
                if "Z" in PHform:
                    Z = PHform["Z"]
                else:
                    Z = projectPSD(E.T @ Q)

            P = PHform["P"]
            S = PHform["S"]
            F = PHform["F"]
        # % Case 2: the matrix Q is provided but not the all other ones
        # %         Solve for these using CVX; see solveJRFPS.m
        else:
            Q = options["PHform"]["Q"]
            if options["standard"] != 1:
                Z = projectPSD(E.T @ Q)
            J, R, F, P, S = solveJRFPS(
                A, E, B, C, D, Q, options["display"], options["weight"]
            )
    else:
        if not "init" in options:
            options["init"] = 1

        # % Case 3: use the 'standard' initialization with Q=I_n and P=0
        if options["init"] == 1:
            Q = np.eye(n)
            if options["standard"] != 1:
                Z = projectPSD(E.T, options["posdef"])

            J = (A - A.T) / 2
            R = projectPSD(J - A, options["posdef"])
            S = (D.T + D) / 2
            S = projectPSD(S, options["posdef"])
            P = np.zeros((n, m))
            F = (B + C.T) / 2
        # % Case 4: use the solution X from the LMIs
        elif options["init"] == 2 or options["init"] == 3:
            [X, delta] = PHformLMIs(sys, options["display"], options["standard"])
            Q = wellcond(X)
            if options["standard"] != 1:
                Z = projectPSD(E.T @ Q, options["posdef"])

            # % J,R,P,S,F computed using the formula for DH form
            if options["init"] == 2:
                [J, R, P, S, F] = PHformJRPSF(X, A, E, B, C, D)
            # % J,R,P,S,F computed using optimization
            elif options["init"] == 3:
                [J, R, F, P, S] = solveJRFPS(A, E, B, C, D, X, options["display"])

        elif options["init"] == 4:
            rng = np.random.default_rng()
            Q = rng.normal(size=n)
            Q = Q @ Q.T
            Q = wellcond(Q, 1e4)
            J = rng.normal(n)
            J = (J - J.T) / 2
            RPS = rng.normal(n + m)
            RPS = projectPSD(RPS, options["posdef"])
            R = RPS[:n, :n]
            P = RPS[:n, n : n + m]
            S = RPS[n : n + m, n : n + m]
            Z = rng.normal(n)
            Z = projectPSD(Z, options["posdef"])
            # % closed-form for F
            QQt = Q @ Q.T
            F = np.linalg.solve((np.eye(n) + QQt), (P + B + Q @ C.T + QQt @ P))

    if not "alpha0" in options:
        options["alpha0"] = 0.5  # Parameter of FGM, can be tuned.
    elif options["alpha0"] <= 0 or options["alpha0"] >= 1:
        logging.error("alpha0 has to be in (0,1).")

    if not "lsparam" in options:
        options["lsparam"] = 1.5
    elif options["lsparam"] < 1:
        logging.error("options.lsparam has to be larger than one for convergence.")

    if not "lsitermax" in options:
        options.lsitermax = 20
    elif options["lsitermax"] < 1:
        logging.error("options.lsitermax has to be larger than one.")
    if not "gradient" in options:
        options["gradient"] = 0

    # % This can happen with the LMI initializations:
    if options["standard"] == 1 and np.min(np.linalg.eigvals(Q)) < 0:
        Q = projectPSD(Q)

    # % Initial step length
    if np.linalg.cond(Q) > 1e6:
        logging.warning("The initial Q is ill-conditioned.")
        logging.warning("You may want to use the function wellcond.py")

    if options["standard"] == 1:
        L = np.linalg.norm(Q, 2) ** 2
    else:
        QQt = Q @ Q.T
        egq = np.linalg.eigvals(QQt)
        L = np.maximum(np.max(egq), 1 / np.min(egq))

    # % Check that E=I for standard systems
    if options["standard"] == 1 and np.linalg.norm(E - np.eye(n), 2) > 1e-6:
        logging.warning("The sought system is standard but E~=I_n")

    # % Initialization
    if options["maxiter"] == np.inf:
        maxiter_for_init = int(1e6)
    else:
        maxiter_for_init = int(options["maxiter"]+1)

    e = np.empty((maxiter_for_init))*np.nan
    t = np.empty((maxiter_for_init))*np.nan
    alpha = np.empty((maxiter_for_init))*np.nan
    beta = np.empty((maxiter_for_init-1))*np.nan # one entry less
    DD2 = (D + D.T) / 2
    if options["standard"] != 1:
        e0, _, _, _, _, _, _, _ = gradnearportHam(
            A, E, B, C, DD2, J, R, Q, Z, P, F, S, options["weight"]
        )
    else:
        e0, _, _, _, _, _, _ = gradnearportHamstd(
            A, B, C, DD2, J, R, Q, P, F, S, options["weight"]
        )
    e[0] = e0

    t[0] = process_time() - cput  # t[0]
    step = 1 / L
    i = 0
    alpha[0] = options["alpha0"]
    Yj = J
    Yr = R
    Yq = Q
    if options["standard"] != 1:
        Yz = Z
    else:
        Z = None
        M = np.eye(n)
    Yp = P
    Yf = F
    Ys = S
    restarti = 1
    if options["display"] == 1:
        print("Display of iteration number and error:")
        # % display parameters
        if n <= 50:
            ndisplay = 100
        elif n <= 100:
            ndisplay = 10
        elif n <= 500:
            ndisplay = 5
        else:
            ndisplay = 1

    while i < options["maxiter"] and t[i] <= options["timemax"]:
        # % Compute gradient
        if options["standard"] != 1:
            _, gJ, gR, gQ, gZ, gP, gF, gS = gradnearportHam(
                A, E, B, C, DD2, Yj, Yr, Yq, Yz, Yp, Yf, Ys, options["weight"]
            )
        else:
            _, gJ, gR, gQ, gP, gF, gS = gradnearportHamstd(
                A, B, C, DD2, Yj, Yr, Yq, Yp, Yf, Ys, options["weight"]
            )
        
        e[i+1] = np.inf
        inneriter = 0
        step = step * 2
        # % Peform line search
        while e[i + 1] > e[i] and (
            (i == 0 and inneriter <= 100) or inneriter <= options["lsitermax"]
        ):
            # % For i == 0, we always have a descent direction
            Jn = Yj - gJ * step
            Jn = (Jn - Jn.T) / 2
            Rn = Yr - gR * step
            Pn = Yp - gP * step
            Fn = Yf - gF * step
            Sn = Ys - gS * step
            # % Project onto K = [R P; P' S] PSD
            RPS = projectPSD(np.block([[Rn, Pn], [Pn.T, Sn]]), options["posdef"])
            Rn = RPS[:n, :n]
            Pn = RPS[:n, n : n + m]
            Sn = RPS[n : n + m, n : n + m]
            Qn = Yq - gQ * step
            if options["standard"] != 1:
                Zn = projectPSD(Yz - gZ * step, options["posdef"])
                e_i, _, _, _, _, _, _, _ = gradnearportHam(
                    A, E, B, C, DD2, Jn, Rn, Qn, Zn, Pn, Fn, Sn, options["weight"]
                )
                e[i + 1] = e_i
            else:
                Qn = projectPSD(Qn)
                e_i, _, _, _, _, _, _ = gradnearportHamstd(
                    A, B, C, DD2, Jn, Rn, Qn, Pn, Fn, Sn, options["weight"]
                )
                e[i + 1] = e_i
            # print(e)
            t[i+1] = process_time() - cput
            step = step / options["lsparam"]
            inneriter += 1
            # print(e)

        if i == 0:
            inneriter0 = inneriter

        # % Conjugate with FGM weights, if decrease was achieved
        # % otherwise restart FGM
        alpha[i+1] = (np.sqrt(alpha[i] ** 4 + 4 * alpha[i] ** 2) - alpha[i] ** 2) / (2)
        beta[i] = alpha[i] * (1 - alpha[i]) / (alpha[i] ** 2 + alpha[i + 1])
        if inneriter >= options["lsitermax"] + 1:  # line search failed
            if restarti == 1:
                # % Restart FGM if not a descent direction
                restarti = 0
                alpha[i + 1] = options["alpha0"]
                Yj = J
                Yr = R
                Yq = Q
                if options["standard"] != 1:
                    Yz = Z
                Yp = P
                Yf = F
                Ys = S
                e[i + 1] = e[i]
                if options["display"] == 1:
                    print(f"Descent could not be achieved: restart. (Iteration {i})", i)
                # % Reinitialize step length
                if options["standard"] == 1:
                    L = np.linalg.norm(Q) ** 2
                else:
                    QQt = Q @ Q.T
                    egq = np.linalg.eigvals(QQt)
                    L = np.maximum(np.max(egq), 1 / np.min(egq))

                # % use the information from the first step: how many update of
                # % step were necessary to obtain decrease
                step = 1 / L / (options["lsparam"] ** inneriter0)

            elif (
                restarti == 0
            ):  # no previous restart and no descent direction => converged
                if options["display"] == 1:
                    print("The algorithm has converged.")

                e[i + 1] = e[i]
                break
        else:
            restarti = 1
            # % Conjugate
            if options["gradient"] == 1:
                beta[i] = 0

            Yj = Jn + beta[i] * (Jn - J)
            Yr = Rn + beta[i] * (Rn - R)
            Yq = Qn + beta[i] * (Qn - Q)
            if options["standard"] != 1:
                Yz = Zn + beta[i] * (Zn - Z)

            Yp = Pn + beta[i] * (Pn - P)
            Yf = Fn + beta[i] * (Fn - F)
            Ys = Sn + beta[i] * (Sn - S)
            # % Keep new iterates in memory
            J = Jn
            R = Rn
            Q = Qn
            if options["standard"] != 1:
                Z = Zn

            P = Pn
            F = Fn
            S = Sn
        # % Display
        if options["display"] == 1:
            if (i+1) % ndisplay == 0:
                print(f"{i}:{e[i+1]} - ")

            if (i+1) % ndisplay * 10 == 0:
                print("\n")
        i += 1
        # % Check if error is small
        if e[i] < 1e-6:
            if options["display"] == 1:
                print("The algorithm has converged.")

            break
    if options["standard"] != 1:
        M = (Z @ np.linalg.solve(Q, np.eye(Q.shape[0]))).T
        R, Z, M, _ = postpro_nearestpencil(R, Z, Q, M)

    if options["display"] == 1:
        print("\n")

    # % Return the system on DH-form (J,R,Q,F,P,S,N,M,Z)
    PHform = {}
    PHform["J"] = J
    PHform["R"] = R
    PHform["Q"] = Q
    PHform["F"] = F
    PHform["P"] = P
    PHform["S"] = S
    PHform["N"] = N
    if options["standard"] != 1:
        PHform["Z"] = Z
        PHform["M"] = M
    else:
        PHform["M"] = np.eye(n)

    phlti_model = PHLTIModel.from_matrices(J= J, R=R,Q=Q, G=F, P=P, S=S,N=N,E=M)

    # remove nans
    e = e[~np.isnan(e)]
    t = t[~np.isnan(t)]

    return phlti_model, e, t


def set_default_options(options):
    default_options = {
        "standard": 0,
        "maxiter": float("inf"),
        "timemax": 10,
        "posdef": 0,
        "PHform": {},
        "init": 1,
        "alpha0": 0.5,
        "lsparam": 1.5,
        "lsitermax": 20,
        "gradient": 0,
        "weight": np.ones(5),
        "display": 1,
    }

    for key, value in default_options.items():
        if key not in options:
            options[key] = value

    return options


def projectPSD(Q, epsilon=0, delta=np.inf):
    # % Project the matrix Q onto the PSD cone
    # %
    # % This requires an eigendecomposition and then setting the negative
    # % eigenvalues to zero,
    # % or all eigenvalues in the interval [epsilon,delta] if specified.

    if Q is None:
        Qp = Q
        return Qp

    Q = (Q + Q.T) / 2
    if np.max(np.max(np.isnan(Q))) == 1 or np.max(np.max(np.isinf(Q))) == 1:
        logging.error("Input matrix has infinite of NaN entries")

    e, V = np.linalg.eig(Q)

    Qp = V @ np.diag(np.minimum(delta, np.maximum(e, epsilon))) @ V.T

    return Qp


def solveJRFPS(A, E, B, C, D, Q, display=1, weights=np.ones((4,1))):
    # % Given Q, find the best (J,R,F,P,S) to approximate (A,E,B,C,D) in DH-form
    # % using CVX:
    # % min_{ J,R,F,P,S } ||A - (J-R)Q||_F + ||B - (F-P)||_F
    # %                      + ||C - (F+P)^T Q||_F + ||(D+D')/2-S||_F
    # % where J = -J' and [R P; P^T S] PSD.
    # %
    # % See the paper 'Finding the nearest positive-real system',
    # % by Nicolas Gillis Punit Sharma, 2017.
    # %
    # % Remark. We do not use the Frobenius norm squared because it requires some
    # % more modelisation work with CVX. Since we refine the solution with FGM,
    # % it does not make a big difference anyway.

    n = A.shape[0]
    m = B.shape[1]

    J = cp.Variable((n, n))
    R = cp.Variable((n, n))
    F = cp.Variable((n, m))
    P = cp.Variable((n, m))
    S = cp.Variable((m, m))

    LMI_matrix = cp.bmat([[R, P], [P.T, S]])

    constraints = []
    constraints += [J == -J.T]
    constraints += [LMI_matrix >> 0]  # Matlab: hermitian_semidefinite( m+n );]
    constraints += [LMI_matrix == LMI_matrix.T]

    minimize = cp.Minimize(
        weights[0] * cp.norm(A - (J - R) @ Q, "fro")
        + weights[1] * cp.norm(B - (F - P), "fro")
        + weights[2] * cp.norm(C - (F + P).T @ Q, "fro")
        + weights[3] * cp.norm(S - (D + D.T) / 2, "fro")
    )

    problem = cp.Problem(minimize, constraints)
    problem.solve()

    return J.value, R.value, F.value, P.value, S.value


def PHformLMIs(sys, display=True, standard=False):
    # % This code allows to find, if possible, a matrix X that solves the LMIs
    # % garanteeing a system (A,E,B,C,D) to be ESPR. If it is not possible, it
    # % provides an solution of a relaxed problem set of LMIs.
    # %
    # % This code uses CVX.
    # %
    # % See the paper 'Finding the nearest positive-real system',
    # % by Nicolas Gillis and Punit Sharma, 2017.
    # %
    # % ****** Input ******
    # % A system sys.(A,E,B,C,D)
    # % display = 1 => display SDPT3 log (default)
    # %
    # % ****** Output ******
    # % delta and X such that the following two matices are PSD and delta^2 is
    # % minimized:
    # %  [-A'*X-X'*A -X'*B+C' ; -B'*X+C D+D'] + delta*eye(n+m)  and
    # %  E'*X + delta*eye(n)

    n = sys["A"].shape[0]
    m = sys["B"].shape[1]

    A = sys["A"]
    B = sys["B"]
    if not "E" in sys or sys["E"] is None:
        E = np.eye(n)
    else:
        E = sys["E"]

    C = sys["C"]
    D = sys["D"]
    # % Solve LMI's (3.2) in our paper, if possible
    # % Otherwise try to satisfy it 'as much as possible'

    X = cp.Variable((n, n))
    delta = cp.Variable(1)
    minimize = cp.Minimize(cp.norm(delta))
    constraints = []
    constraints += [
        cp.bmat([[-A.T @ X - X.T @ A, -X.T @ B + C.T], [-B.T @ X + C, D + D.T]])
        + (delta) * np.eye(n + m)
        >> 0
    ]
    constraints += [E.T @ X + (delta) * np.eye(n) >> 0]
    problem = cp.Problem(minimize, constraints)
    problem.solve()

    X = X.value
    delta = delta.value

    if display == 1:
        if np.linalg.cond(X) <= 1e9 and np.linalg.norm(delta) < 1e-6:
            print("The system admits a DH-form.")
        else:
            print("The system does not admit a DH-form, an approximation was provided.")

    return X, delta


def wellcond(X, delta=1e6):
    # % Given a matrix X, checks whether it is ill-conditioned or rank deficient
    # % and replaces it with a nearby better conditioned matrix.
    # %
    # % delta controls the conditioning of X: X is modified if cond(X) > delta

    if np.linalg.cond(X) > delta:
        logging.warning("X is ill-conditioned: small singular values were increased.")
        u, s, vh = np.linalg.svd(X)
        mini = np.max(s) / delta
        # % Make too small singular value = sigma_max/delta
        Xw = u @ np.diag(np.maximum(mini, s)) @ vh
    else:
        Xw = X
    return Xw


def PHformJRPSF(X, A, E, B, C, D):
    # % Given a matrix X, this function compute the matrices J,R,P,S,F according
    # % to the PH-form of a system
    # %
    # % See the paper 'Finding the nearest positive-real system',
    # % by Nicolas Gillis Punit Sharma, 2017.
    # % In particular; see Theorem 5.
    n = A.shape[0]
    m = B.shape[1]

    X = wellcond(X, 1e6)
    Xm1 = np.linalg.solve(X, np.eye(X.shape[0]))
    AXm1 = A @ Xm1
    J = (AXm1 - AXm1.T) / 2
    R = -(AXm1 + AXm1.T) / 2
    S = (D + D.T) / 2
    F = 0.5 * (B + Xm1 @ C.T)
    P = 0.5 * (-B + Xm1 @ C.T)
    RPS = projectPSD(np.block([[R, P], [P.T, S]]))
    R = RPS[:n, :n]
    P = RPS[:n, n : n + m]
    S = RPS[n : n + m, n : n + m]

    return J, R, P, S, F


def gradnearportHam(A, E, B, C, DD2, J, R, Q, Z, P, F, S, weight):
    # % Value e and gradient (gJ,gR,gQ,gZ,gP,gF,gS) of the objective function
    # %
    # %    weight(1) * || A - (J-R)Q ||_F^2
    # % +  weight(2) * || E' - ZQ^(-1) ||_F^2
    # % +  weight(3) * || B - (F-P)  ||_F^2
    # % +  weight(4) * || C - (F+P)'*Q ||_F^2 +
    # % +  weight(5) * || DD2 - S ||_F^2
    # %
    # % with respect to the variables (J,R,Q,Z,P,F,S).

    Da = (J - R) @ Q - A
    iQ = np.linalg.solve(Q, np.eye(Q.shape[0]))
    ZiQ = Z @ iQ
    De = ZiQ - E.T
    Dbfp = (F - P) - B
    Dcfp = Q.T @ (F + P) - C.T
    Ds = S - DD2
    e = (
        weight[0] * np.linalg.norm(Da, "fro") ** 2
        + weight[1] * np.linalg.norm(Dbfp, "fro") ** 2
        + weight[2] * np.linalg.norm(Dcfp, "fro") ** 2
        + weight[3] * np.linalg.norm(Ds, "fro") ** 2
        + weight[4] * np.linalg.norm(De, "fro") ** 2
    )

    gJ = weight[0] * Da @ Q.T
    gR = -gJ
    gZ = weight[4] * De @ iQ.T
    gQ = (
        weight[0] * (J - R).T @ Da
        - weight[4] * (ZiQ).T @ De @ iQ.T
        + weight[2] * (F + P) @ Dcfp.T
    )
    QDcfp = Q @ (Dcfp)
    gP = weight[1] * -Dbfp + weight[2] * QDcfp
    gF = weight[1] * Dbfp + weight[2] * QDcfp
    gS = weight[3] * Ds

    return e, gJ, gR, gQ, gZ, gP, gF, gS


def gradnearportHamstd(A, B, C, DD2, J, R, Q, P, F, S, weight):
    # % Value (e) and gradient (gJ,gR,gQ,gP,gF,gS) of the objective function
    # %
    # %    weight(1) * || A - (J-R)Q ||_F^2
    # % +  weight(2) * || B - (F-P)  ||_F^2
    # % +  weight(3) * || C - (F+P)'*Q ||_F^2 +
    # % +  weight(4) * || DD2 - S ||_F^2
    # %
    # % with respect to the variables (J,R,Q,P,F,S).

    Da = (J - R) @ Q - A
    Dbfp = (F - P) - B
    Dcfp = Q.T @ (F + P) - C.T
    Ds = S - DD2
    e = (
        weight[0] * np.linalg.norm(Da, "fro") ** 2
        + weight[1] * np.linalg.norm(Dbfp, "fro") ** 2
        + weight[2] * np.linalg.norm(Dcfp, "fro") ** 2
        + weight[3] * np.linalg.norm(Ds, "fro") ** 2
    )

    gJ = weight[0] * Da @ Q.T
    gR = -gJ
    gQ = weight[0] * (J - R).T @ Da + weight[2] * (F + P) @ Dcfp.T
    QDcfp = Q @ (Dcfp)
    gP = weight[1] * -Dbfp + weight[2] * QDcfp
    gF = weight[1] * Dbfp + weight[2] * QDcfp
    gS = weight[3] * Ds

    return e, gJ, gR, gQ, gP, gF, gS


def postpro_nearestpencil(R, Z, Q, M):
    # % Post-processing of (J,R,Q,Z) in the formulation
    # %
    # % min_{J,R,Q,Z} || A - (J-R) * Q ||_F^2 + || E' - ZQ^-1 ||_F^2
    # %
    # % such that J = -J', R and Z PSD.
    # %
    # % Numerically, it may happen than
    # % (1) R and Z are not 'strictly' PSD with very small negative eigenvalues
    # % (2) M^T = ZQ^-1 does not satisfy M^T Q psd (again, very small negative
    # % eigenvalues
    # %
    # % This function postprocess a solution to obtain a nearby feasible one.

    n = len(R)
    # % R and B psd
    Rt, _, _ = strictprojectPSD(R)
    Zt, mineigR, x = strictprojectPSD(Z)

    # % M'*Q psd
    Mt = (Zt @ np.linalg.solve(Q, np.eye(Q.shape[0]))).T
    minMtQ = np.min(np.real(np.linalg.eigvals(Mt.T @ Q)))
    while minMtQ < 0:
        Zt = projectPSD(Zt, 10 ^ (-x))
        Mt = (Zt @ np.linalg.solve(Q, np.eye(Q.shape[0]))).T
        minMtQ = np.min(np.real(np.linalg.eigvals(Mt.T @ Q)))
        x = x - 1

    err = np.array(
        [
            [np.linalg.norm(R - Rt, "fro") / np.linalg.norm(R, "fro")],
            [np.linalg.norm(Z - Zt, "fro") / np.linalg.norm(Z, "fro")],
            [np.linalg.norm(M - Mt, "fro") / np.linalg.norm(M, "fro")],
        ]
    )

    if all(err > 1e-4):
        print(
            f"The postprocessing had to change (Q,R,M): the relative change is, respectively, {err[0]*100}% - {err[1]*100} % - {err[2]*100}%."
        )

    return Rt, Zt, Mt, err


def strictprojectPSD(R):
    # % Given an (almost) PSD matrix, project it onto psd so that it is
    # % numerically PSD (according to the eig function of Matlab)

    R = projectPSD(R, 0)
    mineigR = np.minimum(0, np.min(np.real(np.linalg.eigvals(R))))
    x = 16
    while mineigR < 0:
        R = projectPSD(R, 10 ^ (-x))
        x = x - 1
        mineigR = np.minimum(0, np.min(np.linalg.eigvals(R)))

    return R, mineigR, x
