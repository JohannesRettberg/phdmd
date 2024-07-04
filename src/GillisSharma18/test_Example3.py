import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.io import loadmat
from PHformLMIs import PHformLMIs
from stablePassiveFGM import stablePassiveFGM
from compareINIT import compareINIT
from MSD_DH_system import MSD_DH_system

# Parameters for the experiment
p = 20
n = 2 * p
m = 4

# Noise level: k larger -> noise smaller
k = 4  # In the paper: k=1,2,3,4
epsilon = 2 / (n * k)  # for n=20
# epsilon = 1/(n*k);  # for n=40

# Choose run time for the algorithms
timemax = 1000  # In the paper, we used 300 for n=20, and 1000 for n=40

# If display = 1, logs of the different algorithms are displayed (or not, = 0)
display = 0

# Generate system
At, E, Jt, Rt, Qt = MSD_DH_system(range(1, p+1), range(1, p+1), range(1, p+1))

data_path = "/scratch/tmp/jrettberg/Software/phdmd/src/GillisSharma18/"
if p == 10:  # same 'random' matrices as in the paper
    data = loadmat(os.path.join(data_path,'BCDEx2_204.mat'))
    B = data['B']
    C = data['C']
    D = data['D']
    m = 4
elif p == 20:
    data = loadmat(os.path.join(data_path,'BCDEx2_406.mat'))
    B = data['B']
    C = data['C']
    D = data['D']
    m = 6
else:  # Generate other examples
    B = np.random.rand(2*p, m)
    C = B.T @ Qt
    D = np.random.randn(m, m // 2)
    D = D @ D.T

# Perturbation
N = np.zeros((2*p, 2*p))
N[p:2*p, p:2*p] = -np.eye(p)
Rtn = Rt + epsilon * N
A = At - epsilon * N @ Qt

sys = {'A': A, 'E': E, 'B': B, 'C': C, 'D': D}

# Solve the semidefinite program (5.6)
X, delta = PHformLMIs(sys, display)
print('******* Is the system ESPR? delta^* and stability of (E,A)  **************')
print(f'The optimal value delta^* of the LMIs for this system is      {delta:.4f}.')
print(f'The largest real part of the eigenvalues of the pair (E,A) is {np.max(np.real(eig(A, E)[0])):.4f}.')
print('**************************************************************************')

# Options of the algorithm
options = {'maxiter': np.inf, 'timemax': timemax, 'display': display}

# Compare the three init.: standard, LMIs+formula, LMIs+solve
PHforms, PHforml, PHformo, es, ts, el, tl, eo, to = compareINIT(sys, options)
print('*********************************')
print('4) True initialization :')
print('*********************************')

# 'True' Initialization
PHtrue = {
    'Q': Qt,
    'J': Jt,
    'R': Rt,
    'Z': E.T @ Qt,
    'F': B,
    'P': np.zeros((2*p, m)),
    'S': D
}
options['PHform'] = PHtrue
PHformtrue, ep, tp = stablePassiveFGM(sys, options)
print(f'True init., error = {ep[-1]:.2f}.')

# Display the evolution of the objective function
plt.rc('axes', titlesize=18, labelsize=18)
plt.rc('lines', linewidth=2)

plt.figure()
plt.loglog(tp, ep, linewidth=1.5)
plt.loglog(ts, es, 'r--', linewidth=2.5)
plt.loglog(tl, el, 'g', linewidth=4)
plt.loglog(to, eo, 'k-.', linewidth=2.5)
plt.loglog(to[0], eo[0], 'ko', linewidth=2.5)
plt.title(f'MSD system of size n={2*p}, m={m}')
plt.xlabel('Time (s.)')
plt.ylabel('Error')
plt.axis([0.1, timemax, 0, es[99]])
plt.legend(['FGM - true init.', 'FGM - standard init.', 'FGM - LMIs + formula', 'FGM - LMIs + solve'])
plt.show()
