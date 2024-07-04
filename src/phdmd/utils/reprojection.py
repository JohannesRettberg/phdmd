import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

def myRK4(f, deltaT, x):
    """Runge Kutta 4th order discretization in time."""
    k1 = deltaT * f(x)
    k2 = deltaT * f(x + k1 / 2)
    k3 = deltaT * f(x + k2 / 2)
    k4 = deltaT * f(x + k3)
    return x + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def reproj(nList=None):
    if nList is None:
        nList = [1, 2, 3, 4, 5, 6]

    # Generate system matrix
    N = 10  # full-model dimension
    np.random.seed(1)  # for reproducibility
    D = np.diag(-np.logspace(-1, -2, N))
    W, _ = la.qr(np.random.randn(N, N), mode='economic')
    tcA = W.T @ D @ W  # system matrix of time-continuous system
    np.savetxt('tcA.txt', tcA, fmt='%.16f', delimiter=' ')

    # Discretize with RK4
    deltaT = 1  # time step size
    K = 100  # number of time steps
    f = lambda x: tcA @ x
    makeTimeStepRK4 = lambda x: myRK4(f, deltaT, x)

    # Time-discrete matrix A
    E = np.eye(N)
    A = E + 1/6 * deltaT * (tcA + 2 * tcA @ (E + deltaT / 2 * tcA) + 
        2 * tcA @ (E + deltaT / 2 * (tcA @ (E + deltaT / 2 * tcA))) + 
        tcA @ (E + deltaT * tcA @ (E + deltaT / 2 * (tcA @ (E + deltaT / 2 * tcA)))))

    # Set initial condition
    Xtrain = np.zeros((N, K))
    Xtrain[:, 0] = E[:, 0]
    Xtest = np.zeros((N, K))
    Xtest[:, 0] = E[:, 0] + E[:, 1]

    # Time step full model
    for i in range(1, K):
        Xtrain[:, i] = makeTimeStepRK4(Xtrain[:, i - 1])
        Xtest[:, i] = makeTimeStepRK4(Xtest[:, i - 1])

    for n in nList:
        # Construct basis and project full-model trajectory
        V = E[:, :n]
        XtrainProj = V.T @ Xtrain
        XtestProj = V.T @ Xtest

        # Re-project step
        XtrainReProj = np.zeros((n, K))
        XtrainReProj[:, 0] = V.T @ Xtrain[:, 0]
        for i in range(1, K):
            XtrainReProj[:, i] = V.T @ makeTimeStepRK4(V @ XtrainReProj[:, i - 1])

        # Intrusive model reduction
        Ar = V.T @ A @ V

        # Operator inference without re-proj
        ArOpInf = np.linalg.lstsq(XtrainProj[:, :K-1].T, XtrainProj[:, 1:K].T, rcond=None)[0].T

        # Operator inference with re-proj
        ArOpInfReProj = np.linalg.lstsq(XtrainReProj[:, :K-1].T, XtrainReProj[:, 1:K].T, rcond=None)[0].T

        # Time stepping reduced models
        XtestIntMOR = np.zeros((n, K))
        XtrainIntMOR = np.zeros((n, K))
        XtestIntMOR[:, 0] = V.T @ Xtest[:, 0]
        XtrainIntMOR[:, 0] = V.T @ Xtrain[:, 0]
        XtestOpInf = np.zeros((n, K))
        XtestOpInf[:, 0] = V.T @ Xtest[:, 0]
        XtestOpInfReProj = np.zeros((n, K))
        XtestOpInfReProj[:, 0] = V.T @ Xtest[:, 0]
        for i in range(1, K):
            XtestIntMOR[:, i] = Ar @ XtestIntMOR[:, i - 1]
            XtrainIntMOR[:, i] = Ar @ XtrainIntMOR[:, i - 1]
            XtestOpInf[:, i] = ArOpInf @ XtestOpInf[:, i - 1]
            XtestOpInfReProj[:, i] = ArOpInfReProj @ XtestOpInfReProj[:, i - 1]

        # Plot
        plt.figure()
        plt.plot(np.sqrt(np.sum(XtestProj ** 2, axis=0)), '--g', linewidth=2, label='projected')
        plt.plot(np.sqrt(np.sum(XtestIntMOR ** 2, axis=0)), '-ok', linewidth=2, label='intrusive model reduction')
        plt.plot(np.sqrt(np.sum(XtestOpInf ** 2, axis=0)), '-sr', linewidth=2, label='OpInf, w/out re-proj')
        plt.plot(np.sqrt(np.sum(XtestOpInfReProj ** 2, axis=0)), '-mx', linewidth=2, label='OpInf, re-proj')
        plt.xlabel('time step k')
        plt.ylabel('2-norm of states')
        plt.legend()
        plt.title(f'Dimension n = {n}')
        # plt.axis([-np.inf, np.inf, 0, 1.6])
        plt.show()

if __name__ == "__main__":
    # Example usage
    reproj()