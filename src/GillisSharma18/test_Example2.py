# Numerical experiment from Section 6.2 from the paper
# 'Finding the nearest positive-real system',
# by Nicolas Gillis and Punit Sharma, 2017.
# This example is from the paper
# Wang, Y., Zhang, Z., Koh, C. K., Pang, G. K., & Wong, N.,
# PEDS: Passivity enforcement for descriptor systems via
# hamiltonian-symplectic matrix pencil perturbation. 2010 IEEE/ACM
# Int. Conf. on Computer-Aided Design (ICCAD), pp. 800-807, 2010.

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from compareINIT import compareINIT
from getsystem import getsystem
from PHformLMIs import PHformLMIs


# Define the matrices
A = np.array([[6, -19, 7, -9],
              [11, 3, -21, 18],
              [25, -9, 35, -16],
              [-27, 6, -16, 38]])
B = np.array([-0.6, 1, 0.2, -0.3])[:,np.newaxis]
C = np.array([3.2, 1.4, 2.6, 1.4])[:,np.newaxis].T
D = np.array([0.105])[:,np.newaxis]
E = np.array([[16, 12, -4, 14],
              [14, 8, 4, -14],
              [-14, 8, -4, 34],
              [6, -4, 0, -10]])
# In the second part of the section, we used
# E = np.eye(4);

n, m = B.shape

# Calculate and print the total norm of the system
total_norm = np.linalg.norm(A, 'fro')**2 + np.linalg.norm(B)**2 + np.linalg.norm(C)**2 + np.linalg.norm(D)**2 + np.linalg.norm(E, 'fro')**2
print(f'The total norm of the system is ||(E,A,B,C,D)||_F^2 = {total_norm:.2f}.')

sys = {
    'A': A,
    'E': E,
    'B': B,
    'C': C,
    'D': D
}

X, delta = PHformLMIs(sys, 0)
print('******* Is the system ESPR? delta^* and stability of (E,A)  **************')
print(f'The optimal value delta^* of the LMIs for this system is      {delta:.4f}.')
print(f'The largest real part of the eigenvalues of the pair (E,A) is {max(np.real(la.eigvals(A, E))):.4f}.')
print('**************************************************************************')

# Parameters of the algorithm
options = {
    'maxiter': 999,    # default {np.inf}
    'timemax': np.inf,  # In the paper, we used 120 seconds # default {5}
    'display': 0,
    'weight': np.ones(5)
    # 'weight': [7/4, 7/4, 1/4, 1/4, 1]  # second tested case in the paper
}
# To find the nearest standard system, use
# options['standard'] = 1;

PHforms, PHforml, PHformo, es, ts, el, tl, eo, to = compareINIT(sys, options)

syss = getsystem(PHforms)
sysf = getsystem(PHforml)
sysg = getsystem(PHformo)

# Display results / errors
weight = options["weight"]
print('**************************************************************')
print('     Nearest Positive Real Systems: Results     ')
print('**************************************************************')
print('1) PH-form approximation error for FGM with standard init.:')
print(f' {weight[0]:.2f}*||A-(J-R)Q||_F^2 + {weight[1]:.2f}*||B-(F-P)||_F^2 \n + {weight[2]:.2f}*||C^T-Q^T(F+P)||_F^2 + {weight[3]:.2f}*||D-(S+N)||_F^2  \n + {weight[4]:.2f}*||E-M||_F^2 = {es[-1]:.2f} ')

print('Errors and relative errors :')
print(f'||E-Ep||_F = {np.linalg.norm(sys["E"] - syss["E"], "fro"):.2f}, ||E-Ep||_F/||E||_F = {100 * np.linalg.norm(sys["E"] - syss["E"], "fro") / np.linalg.norm(sys["E"], "fro"):.2f}%')
print(f'||A-Ap||_F = {np.linalg.norm(sys["A"] - syss["A"], "fro"):.2f}, ||A-Ap||_F/||A||_F = {100 * np.linalg.norm(sys["A"] - syss["A"], "fro") / np.linalg.norm(sys["A"], "fro"):.2f}%')
print(f'||B-Bp||_F = {np.linalg.norm(sys["B"] - syss["B"], "fro"):.2f}, ||B-Bp||_F/||B||_F = {100 * np.linalg.norm(sys["B"] - syss["B"], "fro") / np.linalg.norm(sys["B"], "fro"):.2f}%')
print(f'||C-Cp||_F = {np.linalg.norm(sys["C"] - syss["C"], "fro"):.2f}, ||C-Cp||_F/||C||_F = {100 * np.linalg.norm(sys["C"] - syss["C"], "fro") / np.linalg.norm(sys["C"], "fro"):.2f}%')
print(f'||D-Dp||_F = {np.linalg.norm(sys["D"] - syss["D"], "fro"):.2f}, ||D-Dp||_F/||D||_F = {100 * np.linalg.norm(sys["D"] - syss["D"], "fro") / np.linalg.norm(sys["D"], "fro"):.2f}%')

print('------------------------------------------------------------')
print('2) PH-form approximation error for FGM with LMIs+formula :')
print(f' {weight[0]:.2f}*||A-(J-R)Q||_F^2 + {weight[1]:.2f}*||B-(F-P)||_F^2 \n + {weight[2]:.2f}*||C^T-Q^T(F+P)||_F^2 + {weight[3]:.2f}*||D-(S+N)||_F^2  \n + {weight[4]:.2f}*||E-M||_F^2 = {el[-1]:.2f} ')

print('Errors and relative errors :')
print(f'||E-Ep||_F = {np.linalg.norm(sys["E"] - sysf["E"], "fro"):.2f}, ||E-Ep||_F/||E||_F = {100 * np.linalg.norm(sys["E"] - sysf["E"], "fro") / np.linalg.norm(sys["E"], "fro"):.2f}%')
print(f'||A-Ap||_F = {np.linalg.norm(sys["A"] - sysf["A"], "fro"):.2f}, ||A-Ap||_F/||A||_F = {100 * np.linalg.norm(sys["A"] - sysf["A"], "fro") / np.linalg.norm(sys["A"], "fro"):.2f}%')
print(f'||B-Bp||_F = {np.linalg.norm(sys["B"] - sysf["B"], "fro"):.2f}, ||B-Bp||_F/||B||_F = {100 * np.linalg.norm(sys["B"] - sysf["B"], "fro") / np.linalg.norm(sys["B"], "fro"):.2f}%')
print(f'||C-Cp||_F = {np.linalg.norm(sys["C"] - sysf["C"], "fro"):.2f}, ||C-Cp||_F/||C||_F = {100 * np.linalg.norm(sys["C"] - sysf["C"], "fro") / np.linalg.norm(sys["C"], "fro"):.2f}%')
print(f'||D-Dp||_F = {np.linalg.norm(sys["D"] - sysf["D"], "fro"):.2f}, ||D-Dp||_F/||D||_F = {100 * np.linalg.norm(sys["D"] - sysf["D"], "fro") / np.linalg.norm(sys["D"], "fro"):.2f}%')

print('------------------------------------------------------------')
print('3) PH-form approximation error for FGM with LMIs+solve :')
print(f' {weight[0]:.2f}*||A-(J-R)Q||_F^2 + {weight[1]:.2f}*||B-(F-P)||_F^2 \n + {weight[2]:.2f}*||C^T-Q^T(F+P)||_F^2 + {weight[3]:.2f}*||D-(S+N)||_F^2 \n + {weight[4]:.2f}*||E-M||_F^2 = {eo[-1]:.2f} ')

print('Errors and relative errors :')
print(f'||E-Epg||_F = {np.linalg.norm(sys["E"] - sysg["E"], "fro"):.2f}, ||E-Epg||_F/||E||_F = {100 * np.linalg.norm(sys["E"] - sysg["E"], "fro") / np.linalg.norm(sys["E"], "fro"):.2f}%')
print(f'||A-Apg||_F = {np.linalg.norm(sys["A"] - sysg["A"], "fro"):.2f}, ||A-Apg||_F/||A||_F = {100 * np.linalg.norm(sys["A"] - sysg["A"], "fro") / np.linalg.norm(sys["A"], "fro"):.2f}%')
print(f'||B-Bpg||_F = {np.linalg.norm(sys["B"] - sysg["B"], "fro"):.2f}, ||B-Bpg||_F/||B||_F = {100 * np.linalg.norm(sys["B"] - sysg["B"], "fro") / np.linalg.norm(sys["B"], "fro"):.2f}%')
print(f'||C-Cpg||_F = {np.linalg.norm(sys["C"] - sysg["C"], "fro"):.2f}, ||C-Cpg||_F/||C||_F = {100 * np.linalg.norm(sys["C"] - sysg["C"], "fro") / np.linalg.norm(sys["C"], "fro"):.2f}%')
print(f'||D-Dpg||_F = {np.linalg.norm(sys["D"] - sysg["D"], "fro"):.2f}, ||D-Dpg||_F/||D||_F = {100 * np.linalg.norm(sys["D"] - sysg["D"], "fro") / np.linalg.norm(sys["D"], "fro"):.2f}%')

# Display evolution of the objective function
plt.rc('axes', titlesize=18, labelsize=18)
plt.rc('lines', linewidth=2)

plt.figure()
plt.loglog(ts, es, 'r--', linewidth=2.5)
plt.loglog(tl, el, 'g', linewidth=4)
plt.loglog(to, eo, 'k-.', linewidth=2.5)
plt.loglog(to[0], eo[0], 'ko', linewidth=2.5)
plt.title(f'System of size n={n}, m={m}')
plt.xlabel('Time (s.)')
plt.ylabel('Error')
plt.axis([0, options["timemax"], 0, es[0]])
plt.legend(['FGM - standard init.', 'FGM - LMIs + formula', 'FGM - LMIs + solve'])
plt.show()