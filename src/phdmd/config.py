import os

import numpy as np
import matplotlib as mpl

from scipy.signal import sawtooth
from pymor.models.iosys import PHLTIModel, LTIModel

from phdmd.algorithm.methods import IODMDMethod, PHDMDMethod, OIMethod, PHDMDHMethod
from phdmd.model.msd import msd
from phdmd.model.poro import poro


class Experiment:
    """
    Class for experiments.

    Parameters
    ----------
    name: str
        The name of the experiment.
    model: str
        The name of the model of the experiment.
    fom : function
        Function for getting port-Hamiltonian system matrices.
    u : function
        Training input function.
    T : numpy.ndarray
        Training time interval.
    x0 : numpy.ndarray
        Training initial condition.
    time_stepper : str, optional
        Name of the time stepping method. Default 'implicit_midpoint'.

    """

    def __init__(self, name, model, fom, u, T, x0, time_stepper='implicit_midpoint', u_test=None, T_test=None,
                 x0_test=None, r=None, noise=None, methods=None, use_Berlin=True, HQ_init_strat = 'known', perturb_energy_matrix = False,
                 use_Riccatti_transform = False, perturb_value = None):
        if methods is None:
            methods = [PHDMDMethod()]

        self.name = name
        self.model = model
        self.use_Berlin = use_Berlin
        self.HQ_init_strat = HQ_init_strat
        self.perturb_energy_matrix = perturb_energy_matrix
        self.use_Riccatti_transform = use_Riccatti_transform
        self.perturb_value = perturb_value

        H, J, R, G, P, S, N, Q = fom()
        self.fom = PHLTIModel.from_matrices(J, R, G, P, S, N, H, Q=Q)

        self.H = H
        self.Q = Q

        self.u = u
        self.T = T
        self.x0 = x0
        self.delta = T[1] - T[0]
        self.time_stepper = time_stepper

        if u_test is None:
            u_test = u

        if T_test is None:
            T_test = T

        if x0_test is None:
            x0_test = x0

        self.u_test = u_test
        self.T_test = T_test
        self.x0_test = x0_test

        self.r = r
        self.noise = noise

        self.methods = methods

# choose if Berlin form (True) or pH standard with Q (False)
use_Berlin = True   # (bool)
HQ_init_strat = 'known' # 'id' | 'rand' | 'known'
# perturbation
perturb_energy_matrix = True # HQ_init_strat needs to be 'known'
perturb_value = 1e-8 #'last_sing' # (float) or 'last_sing' for last singular value
# Riccatti transform
use_Riccatti_transform = False

if perturb_energy_matrix:
    assert HQ_init_strat == 'known'

siso_msd_exp = Experiment(
    name='SISO_MSD',
    model='msd',
    fom=lambda: msd(6, 1, use_Berlin=use_Berlin),
    u=lambda t: np.array([np.exp(-0.5 * t) * np.sin(t ** 2)]),
    T=np.linspace(0, 4, 101),
    x0=np.zeros(6),
    u_test=lambda t: np.array([sawtooth(2 * np.pi * 0.5 * t)]),
    T_test=np.linspace(0, 10, 251),
    x0_test=np.zeros(6),
    methods=[PHDMDHMethod()],
    use_Berlin = use_Berlin,
    HQ_init_strat = HQ_init_strat,
    perturb_energy_matrix = perturb_energy_matrix,
    use_Riccatti_transform = use_Riccatti_transform,
    perturb_value = perturb_value,
)

# siso_msd_exp_1 = Experiment(
#     name='SISO_MSD',
#     model='msd',
#     fom=lambda: msd(6, 1),
#     u=lambda t: np.array([np.exp(-0.5 * t) * np.sin(t ** 2)]),
#     T=np.linspace(0, 4, 101),
#     x0=np.zeros(6),
#     u_test=lambda t: np.array([sawtooth(2 * np.pi * 0.5 * t)]),
#     T_test=np.linspace(0, 10, 251),
#     x0_test=np.zeros(6),
#     methods=[IODMDMethod(), PHDMDMethod()],
# )

# siso_msd_exp_2 = Experiment(
#     name='SISO_MSD_small_delta',
#     model='msd',
#     fom=lambda: msd(6, 1),
#     u=lambda t: np.array([np.exp(-0.5 * t) * np.sin(t ** 2)]),
#     T=np.linspace(0, 4, 40001),
#     x0=np.zeros(6),
#     u_test=lambda t: np.array([sawtooth(2 * np.pi * 0.5 * t)]),
#     T_test=np.linspace(0, 10, 100001),
#     x0_test=np.zeros(6),
#     methods=[IODMDMethod(), PHDMDMethod()],
# )

# siso_msd_exp_3 = Experiment(
#     name='SISO_MSD_RK45',
#     model='msd',
#     fom=lambda: msd(6, 1),
#     u=lambda t: np.array([np.exp(-0.5 * t) * np.sin(t ** 2)]),
#     T=np.linspace(0, 4, 101),
#     x0=np.zeros(6),
#     time_stepper='RK45',
#     u_test=lambda t: np.array([sawtooth(2 * np.pi * 0.5 * t)]),
#     T_test=np.linspace(0, 10, 251),
#     x0_test=np.zeros(6),
#     methods=[IODMDMethod(), PHDMDMethod()],
# )

# siso_msd_exp_4 = Experiment(
#     name='SISO_MSD_noisy',
#     model='msd',
#     fom=lambda: msd(6, 1),
#     u=lambda t: np.array([np.exp(-0.5 * t) * np.sin(t ** 2)]),
#     T=np.linspace(0, 4, 101),
#     x0=np.zeros(6),
#     u_test=lambda t: np.array([sawtooth(2 * np.pi * 0.5 * t)]),
#     T_test=np.linspace(0, 10, 251),
#     x0_test=np.zeros(6),
#     noise=1e-4,
#     methods=[OIMethod(), PHDMDMethod()],
# )

mimo_msd_exp = Experiment(
    name='MIMO_MSD',
    model='msd',
    fom=lambda: msd(100, 2),
    u=lambda t: np.array([np.exp(-0.5 / 100 * t) * np.sin(1 / 100 * t ** 2),
                          np.exp(-0.5 / 100 * t) * np.cos(1 / 100 * t ** 2)]),
    T=np.linspace(0, 4 * 100, 100 * 100 + 1),
    x0=np.zeros(100),
    u_test=lambda t: np.array([sawtooth(2 * np.pi * 0.5 * t), -sawtooth(2 * np.pi * 0.5 * t)]),
    T_test=np.linspace(0, 10, 251),
    x0_test=np.zeros(100),
    methods=[PHDMDMethod()], # [OIMethod(), PHDMDMethod()]
    use_Berlin = use_Berlin,
    HQ_init_strat = HQ_init_strat,
    perturb_energy_matrix = perturb_energy_matrix,
    use_Riccatti_transform = use_Riccatti_transform,
    perturb_value = perturb_value,
)

# poro_exp = Experiment(
#     name='PORO',
#     model='poro',
#     fom=lambda: poro(980),
#     u=lambda t: np.array([np.exp(-0.5 / 100 * t) * np.sin(1 / 100 * t ** 2),
#                           np.exp(-0.5 / 100 * t) * np.cos(1 / 100 * t ** 2)]),
#     T=np.linspace(0, 4 * 100, 100 * 100 + 1),
#     x0=np.zeros(980),
#     u_test=lambda t: np.array([sawtooth(2 * np.pi * 0.5 * t), -sawtooth(2 * np.pi * 0.5 * t)]),
#     T_test=np.linspace(0, 10, 251),
#     x0_test=np.zeros(980)
# )

# experiments = [siso_msd_exp, siso_msd_exp_1, siso_msd_exp_2, siso_msd_exp_3, siso_msd_exp_4]
# experiments = [mimo_msd_exp]
# experiments = [poro_exp]
experiments = [siso_msd_exp]

save_results = True  # If true all figures will be saved as pdf
width_pt = 420  # Get this from LaTeX using \the\textwidth
fraction = 0.49 if save_results else 1  # Fraction of width the figure will occupy
plot_format = 'pdf'

colors = np.array(mpl.colormaps['Set1'].colors)
# delete yello
colors = np.delete(colors,(5),axis=0)

# create relative paths from config file
working_dir = os.path.dirname(__file__)


data_path = os.path.join(working_dir,'../data')
simulations_path = os.path.join(data_path, 'simulations_240515')
evaluations_path = os.path.join(data_path, 'evaluations_240515')
plots_path = os.path.join(evaluations_path,'plots')

for path in [data_path, simulations_path, evaluations_path, plots_path]:
    if not os.path.exists(path):
        os.makedirs(path)

force_simulation = True  # If true the simulation will be forced to run again
