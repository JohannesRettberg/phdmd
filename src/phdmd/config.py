import os

import numpy as np
import matplotlib as mpl

from scipy.signal import sawtooth
from pymor.models.iosys import PHLTIModel, LTIModel

from phdmd.algorithm.methods import (
    IODMDMethod,
    PHDMDMethod,
    OIMethod,
    PHDMDHMethod,
    CVXABCDPHMethod,
    CVXABCDPRMethod,
    CVXPHMethod,
)
from phdmd.model.msd import msd
from phdmd.model.msd_rettberg import create_mass_spring_damper_system
from phdmd.model.poro import poro
from phdmd.utils.gillis_options import GillisOptions

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

    def __init__(
        self,
        name,
        model,
        fom,
        u,
        T,
        x0,
        time_stepper="implicit_midpoint",
        u_test=None,
        T_test=None,
        x0_test=None,
        r=None,
        noise=None,
        methods=None,
        use_Berlin=True,
        HQ_init_strat="known",
        perturb_energy_matrix=False,
        use_Riccatti_transform=False,
        perturb_value=None,
        use_cholesky_like_fact=False,
        use_cvx=False,
        use_known_J=False,
        use_projection_of_A=False,
        constraint_type="no",
        gillis_options = None,
    ):
        if methods is None:
            methods = [PHDMDMethod()]

        self.name = name
        self.model = model
        self.use_Berlin = use_Berlin
        self.HQ_init_strat = HQ_init_strat
        self.perturb_energy_matrix = perturb_energy_matrix
        self.use_Riccatti_transform = use_Riccatti_transform
        self.perturb_value = perturb_value
        self.use_cholesky_like_fact = use_cholesky_like_fact
        self.use_cvx = use_cvx
        self.use_known_J = use_known_J
        self.use_projection_of_A = use_projection_of_A
        self.constraint_type = constraint_type
        self.gillis_options = gillis_options

        H, J, R, G, P, S, N, Q = fom()
        self.fom = PHLTIModel.from_matrices(J, R, G, P, S, N, E=H, Q=Q)

        self.H = H
        self.Q = Q
        self.J = J

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
use_Berlin = True  # (bool)
HQ_init_strat = "known"  # 'id' | 'rand' | 'known'
# perturbation
perturb_energy_matrix = False  # HQ_init_strat needs to be 'known'
perturb_value = 1e-4  #'last_sing' # (float) or 'last_sing' for last singular value
# Riccatti transform
use_Riccatti_transform = False
# Cholesky-like factorization to Poisson form (so far only for use_Berlin implemented)
use_cholesky_like_fact = False
# choose if cvx should be used for skew-symmetric identification of J instead of analytic Procrustes
# additionally choose if known J should be included
use_cvx = False
use_known_J = False

# %% for the OI of a state space system
use_projection_of_A = False  # project a on snd

# for ABCD inferring with cvx, choose constraints (mostly on A)
constraint_type = "nsd"  # "KYP"| "KYP_relaxed" | "no" | "nsd" | "nsd"

# for CVXABCDPR
options = {"standard": 1, "init": 3}
gillis_options = GillisOptions(options) 


if perturb_energy_matrix:
    assert HQ_init_strat == "known" or (
        HQ_init_strat == "id" and use_Riccatti_transform
    )

siso_msd_exp = Experiment(
    name="SISO_MSD",
    model="msd",
    fom=lambda: msd(6, 1, use_Berlin=use_Berlin),
    u=lambda t: np.array([np.exp(-0.5 * t) * np.sin(t**2)]),
    T=np.linspace(0, 4, 101),
    x0=np.zeros(6),
    u_test=lambda t: np.array([sawtooth(2 * np.pi * 0.5 * t)]),
    T_test=np.linspace(0, 10, 251),
    x0_test=np.zeros(6),
    methods=[PHDMDHMethod()],
    use_Berlin=use_Berlin,
    HQ_init_strat=HQ_init_strat,
    perturb_energy_matrix=perturb_energy_matrix,
    use_Riccatti_transform=use_Riccatti_transform,
    perturb_value=perturb_value,
    use_cholesky_like_fact=use_cholesky_like_fact,
    use_cvx=use_cvx,
    use_known_J=use_known_J,
    use_projection_of_A=use_projection_of_A,
)

siso_msd_exp_RG = Experiment(
    name="SISO_MSD_RG",
    model="msd",
    fom=lambda: create_mass_spring_damper_system(
        n_mass=3,
        mass_vals=4,
        damp_vals=1,
        stiff_vals=4,
        input_vals=0,
        system="ph",
        use_Berlin=use_Berlin,
        use_cholesky_like_fact=use_cholesky_like_fact,
    ),
    u=lambda t: np.array([np.exp(-0.5 * t) * np.sin(t**2)]),
    T=np.linspace(0, 4, 101),
    x0=np.zeros(6),
    u_test=lambda t: np.array([sawtooth(2 * np.pi * 0.5 * t)]),
    T_test=np.linspace(0, 10, 251),
    x0_test=np.zeros(6),
    methods=[CVXABCDPRMethod()],
    use_Berlin=use_Berlin,
    HQ_init_strat=HQ_init_strat,
    perturb_energy_matrix=perturb_energy_matrix,
    use_Riccatti_transform=use_Riccatti_transform,
    perturb_value=perturb_value,
    use_cholesky_like_fact=use_cholesky_like_fact,
    use_cvx=use_cvx,
    use_known_J=use_known_J,
    use_projection_of_A=use_projection_of_A,
    constraint_type = constraint_type,
    gillis_options = gillis_options,
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
# methods=[OIMethod(), PHDMDMethod()],
# )

mimo_msd_exp = Experiment(
    name="MIMO_MSD",
    model="msd",
    fom=lambda: msd(100, 2),
    u=lambda t: np.array(
        [
            np.exp(-0.5 / 100 * t) * np.sin(1 / 100 * t**2),
            np.exp(-0.5 / 100 * t) * np.cos(1 / 100 * t**2),
        ]
    ),
    T=np.linspace(0, 4 * 100, 100 * 100 + 1),
    x0=np.zeros(100),
    u_test=lambda t: np.array(
        [sawtooth(2 * np.pi * 0.5 * t), -sawtooth(2 * np.pi * 0.5 * t)]
    ),
    T_test=np.linspace(0, 10, 251),
    x0_test=np.zeros(100),
    methods=[
        # CVXABCDPHMethod(),
        CVXABCDPRMethod(),
    ],  # [OIMethod(), PHDMDMethod()] CVXPHMethod(), CVXABCDMethod()
    use_Berlin=use_Berlin,
    HQ_init_strat=HQ_init_strat,
    perturb_energy_matrix=perturb_energy_matrix,
    use_Riccatti_transform=use_Riccatti_transform,
    perturb_value=perturb_value,
    use_cholesky_like_fact=use_cholesky_like_fact,
    use_cvx=use_cvx,
    use_known_J=use_known_J,
    use_projection_of_A=use_projection_of_A,
    constraint_type=constraint_type,
    gillis_options = gillis_options,
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
experiments = [mimo_msd_exp]
# experiments = [poro_exp]
# experiments = [siso_msd_exp_RG]

save_results = True  # If true all figures will be saved as pdf
width_pt = 420  # Get this from LaTeX using \the\textwidth
fraction = 0.49 if save_results else 1  # Fraction of width the figure will occupy
plot_format = "pdf"

colors = np.array(mpl.colormaps["Set1"].colors)
# delete yello
colors = np.delete(colors, (5), axis=0)

# create relative paths from config file
working_dir = os.path.dirname(__file__)


data_path = os.path.join(working_dir, "../data")
date_meeting = "20240702"
path_ending = "cvxabcdpr_firsttry"
simulations_path = os.path.join(data_path, f"simulations_{date_meeting}_{path_ending}")
evaluations_path = os.path.join(data_path, f"evaluations_{date_meeting}_{path_ending}")
plots_path = os.path.join(evaluations_path, "plots")

for path in [data_path, simulations_path, evaluations_path, plots_path]:
    if not os.path.exists(path):
        os.makedirs(path)

force_simulation = True  # If true the simulation will be forced to run again
