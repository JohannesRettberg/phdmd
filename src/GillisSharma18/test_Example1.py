import numpy as np
from scipy.linalg import inv, eig
import matplotlib.pyplot as plt

from stablePassiveFGM import stablePassiveFGM
from getsystem import getsystem

# % Numerical experiment from Section 6.1 from the paper
# % 'Finding the nearest positive-real system',
# % by Nicolas Gillis and Punit Sharma, 2017.
# % This example is from the paper
# % N. Guglielmi, D. Kressner, and C. Lubich, Low rank differential equations
# % for Hamiltonian matrix nearness problems, Numerische Mathematik,
# % 129 (2015), pp. 279-319.

sys = {}
sys["A"] = np.array(
    [[-0.08, 0.83, 0, 0], [-0.83, -0.08, 0, 0], [0, 0, -0.7, 9], [0, 0, -9, -0.7]]
)
sys["B"] = np.array([[1, 1], [0, 0], [1, -1], [0, 0]])
sys["C"] = np.array([[0.4, 0.6], [0, 0], [0.4, 1], [0, 0]]).T
sys["D"] = np.array([[0.3, 0], [0, -0.15]])
sys["E"] = np.eye(4)

# Check the second condition for the system to be PR
s = 1 + 2j
G = sys["C"].dot(inv(s * np.eye(sys["A"].shape[0]) - sys["A"])).dot(sys["B"]) + sys["D"]
min_eig_val = np.min(np.linalg.eigvals(G + G.conj().T).real)
print(
    f"The second condition for the system to be PR is not satisfied \nfor s = {s.real}+{s.imag}j because min(eig(G+G*)) = {min_eig_val:.2f}."
)

# Parameters of the algorithm
options = {
    "maxiter": 299,
    "timemax": 2,  # In the paper, we used 5 seconds
    "display": 0,
}

# Weights for the objective function
# weight = np.ones(5)
# % Second tested case in the paper: 
weight = np.array([7/4, 7/4, 1/4, 1/4, 1]) 
options["weight"] = weight

# To find the nearest standard system
options["standard"] = 1

# Initialization
options["init"] = 1  # Standard initialization

# Running the algorithms
print("******  Running FGM   ********")
PHform, e, t = stablePassiveFGM(
    sys, options
)  # Define this function as per your algorithm
sysf = getsystem(PHform)  # Define this function as per your algorithm

print("****  Running Gradient   ********")
options["gradient"] = 1
PHformg, eg, tg = stablePassiveFGM(
    sys, options
) 
sysg = getsystem(PHformg)  # Define this function as per your algorithm

# Display results / errors
print("**************************************************************")
print("     Nearest Positive Real Systems: Results       ")
print("**************************************************************")
print("1) PH-form approximation error for FGM :")
print(
    f" {weight[0]:.2f}*||A-(J-R)Q||_F^2 + {weight[1]:.2f}*||B-(F-P)||_F^2 \n + {weight[2]:.2f}*||C^T-Q^T(F+P)||_F^2 + {weight[3]:.2f}*||D-(S+N)||_F^2  \n + {weight[4]:.2f}*||E-M||_F^2 = {e[-1]:.2f}"
)

print("Errors and relative errors :")
print(
    f'||E-Ep||_F = {np.linalg.norm(sys["E"] - sysf["E"], "fro"):.2f}, ||E-Ep||_F/||E||_F = {100 * np.linalg.norm(sys["E"] - sysf["E"], "fro") / np.linalg.norm(sys["E"], "fro"):.2f}%'
)
print(
    f'||A-Ap||_F = {np.linalg.norm(sys["A"] - sysf["A"], "fro"):.2f}, ||A-Ap||_F/||A||_F = {100 * np.linalg.norm(sys["A"] - sysf["A"], "fro") / np.linalg.norm(sys["A"], "fro"):.2f}%'
)
print(
    f'||B-Bp||_F = {np.linalg.norm(sys["B"] - sysf["B"], "fro"):.2f}, ||B-Bp||_F/||B||_F = {100 * np.linalg.norm(sys["B"] - sysf["B"], "fro") / np.linalg.norm(sys["B"], "fro"):.2f}%'
)
print(
    f'||C-Cp||_F = {np.linalg.norm(sys["C"] - sysf["C"], "fro"):.2f}, ||C-Cp||_F/||C||_F = {100 * np.linalg.norm(sys["C"] - sysf["C"], "fro") / np.linalg.norm(sys["C"], "fro"):.2f}%'
)
print(
    f'||D-Dp||_F = {np.linalg.norm(sys["D"] - sysf["D"], "fro"):.2f}, ||D-Dp||_F/||D||_F = {100 * np.linalg.norm(sys["D"] - sysf["D"], "fro") / np.linalg.norm(sys["D"], "fro"):.2f}%'
)

print("------------------------------------------------------------")
print("2) PH-form approximation error for Gradient :")
print(
    f" {weight[0]:.2f}*||A-(J-R)Q||_F^2 + {weight[1]:.2f}*||B-(F-P)||_F^2 \n + {weight[2]:.2f}*||C^T-Q^T(F+P)||_F^2 + {weight[3]:.2f}*||D-(S+N)||_F^2 \n + {weight[4]:.2f}*||E-M||_F^2 = {eg[-1]:.2f}"
)

print("Errors and relative errors :")
print(
    f'||E-Epg||_F = {np.linalg.norm(sys["E"] - sysg["E"], "fro"):.2f}, ||E-Epg||_F/||E||_F = {100 * np.linalg.norm(sys["E"] - sysg["E"], "fro") / np.linalg.norm(sys["E"], "fro"):.2f}%'
)
print(
    f'||A-Apg||_F = {np.linalg.norm(sys["A"] - sysg["A"], "fro"):.2f}, ||A-Apg||_F/||A||_F = {100 * np.linalg.norm(sys["A"] - sysg["A"], "fro") / np.linalg.norm(sys["A"], "fro"):.2f}%'
)
print(
    f'||B-Bpg||_F = {np.linalg.norm(sys["B"] - sysg["B"], "fro"):.2f}, ||B-Bpg||_F/||B||_F = {100 * np.linalg.norm(sys["B"] - sysg["B"], "fro") / np.linalg.norm(sys["B"], "fro"):.2f}%'
)
print(
    f'||C-Cpg||_F = {np.linalg.norm(sys["C"] - sysg["C"], "fro"):.2f}, ||C-Cpg||_F/||C||_F = {100 * np.linalg.norm(sys["C"] - sysg["C"], "fro") / np.linalg.norm(sys["C"], "fro"):.2f}%'
)
print(
    f'||D-Dpg||_F = {np.linalg.norm(sys["D"] - sysg["D"], "fro"):.2f}, ||D-Dpg||_F/||D||_F = {100 * np.linalg.norm(sys["D"] - sysg["D"], "fro") / np.linalg.norm(sys["D"], "fro"):.2f}%'
)

# tranform to array
t = np.array(t)


# Display evolution of the objective function
plt.figure()
plt.plot(t, e, label="FGM")
plt.plot(tg, eg, "r--", label="Gradient")
plt.title("General PR system")
plt.xlabel("Time (s)")
plt.ylabel("Error")
plt.legend()
plt.show(block=False)

print('breakpoint in test_Example1')
