import logging
import os
import numpy as np
from matplotlib import pyplot as plt
import scipy.linalg as la

from phdmd import config
from phdmd.data.generate import generate
from phdmd.evaluation.evaluation import evaluate
from pymor.algorithms.to_matrix import to_matrix
from phdmd.linalg.definiteness import project_spsd, project_spd, project_snd
from phdmd.utils.preprocess_data import (
    get_Riccati_transform,
    get_initial_energy_matrix,
    perturb_initial_energy_matrix,
)
from phdmd.utils.postprocess_data import compare_matrices
from phdmd.data.generate import sim

import cProfile
import pstats
from pymor.models.iosys import PHLTIModel, LTIModel

import opinf


def main():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    if not os.path.exists(config.simulations_path):
        os.makedirs(config.simulations_path)

    if not os.path.exists(config.evaluations_path):
        os.makedirs(config.evaluations_path)

    # specify experiments in config.py
    experiments = config.experiments

    for exp in experiments:
        logging.info(f"Experiment: {exp.name}")
        lti_dict = {}

        if exp.use_Riccatti_transform:
            T = get_Riccati_transform(exp.fom)

        n = exp.fom.order
        use_Berlin = exp.use_Berlin

        H, Q = get_initial_energy_matrix(exp)

        if exp.perturb_energy_matrix:
            H, Q = perturb_initial_energy_matrix(exp, H, Q)

        if exp.use_known_J:
            assert exp.use_cvx
            J_known = exp.J
        else:
            J_known = None

        logging.info(f"State dimension n = {n}")
        logging.info(f"Step size delta = {exp.delta:.2e}")
        lti_dict["Original"] = exp.fom

        # Generate/Load training data
        X_train, Y_train, U_train = generate(exp)
        # plot_state_over_time(exp.T, X_train)

        if exp.use_Riccatti_transform:
            X_train = T @ X_train

        use_residual_energy_pod_treshold = False
        if use_residual_energy_pod_treshold:
            # Compute the POD basis, using the residual energy decay to select r.
            basis = opinf.basis.PODBasis(residual_energy=1e-8).fit(X_train)
            # Check the decay of the singular values and the associated residual energy.
            basis.plot_energy(right=25)
            plt.savefig(f"energy_of_POD.png")
        else:
            # define basis by number of basis vectors
            red_dim = 24
            basis = opinf.basis.PODBasis(num_vectors=red_dim).fit(X_train)

        # Instantiate the model.
        rom = opinf.models.ContinuousModel(
            operators=[
                opinf.operators.LinearOperator(),
                opinf.operators.InputOperator(),
            ]
        )

        # Compress the snapshot data.
        X_train_comp = basis.compress(X_train)

        # Estimate time derivatives (dq/dt) for each training snapshot.
        dt = exp.T[1] - exp.T[0]
        X_train_comp, Xdot_train_comp, U_train_fit = opinf.ddt.bwd1(
            X_train_comp, dt, U_train
        )
        # workaround to also compress outputs
        _, _, Y_train_fit = opinf.ddt.bwd1(X_train_comp, dt, Y_train)

        # Train the reduced-order model.
        logging.info(f"Fitting operators to data...")
        rom.fit(states=X_train_comp, ddts=Xdot_train_comp, inputs=U_train_fit)
        print(rom)

        # Express the initial condition in the coordinates of the basis.
        x0_ = basis.compress(exp.x0)

        # project A on snd
        if exp.use_projection_of_A:
            A_OI = project_snd(rom.A_.entries)
        else:
            A_OI = rom.A_.entries

        # check neg. def
        logging.info(f"Checking eigenvalues...")
        eig_vals_A = np.linalg.eigvals(A_OI)
        logging.info(
            f"Maximum real-part of eigenvalue is negative: {all(np.real(eig_vals_A)<0)}. Max. EV is {np.max(np.real(eig_vals_A))}."
        )

        # Solve the reduced-order model using Implicit Euler.
        x_red = implicit_euler(exp.T, x0_, A_OI, rom.B_.entries, np.squeeze(U_train))
        X_ROM = basis.decompress(x_red)

        plot_trajectory_over_time(exp.T, X_train, X_ROM)

        logging.info(f"Calculate errors...")
        rel_froerr_projection = basis.projection_error(X_train, relative=True)
        rel_froerr_opinf = opinf.post.frobenius_error(X_train, X_ROM)[1]

        print(
            "Relative Frobenius-norm errors",
            "-" * 33,
            f"projection error:\t{rel_froerr_projection:%}",
            f"OpInf ROM error:\t{rel_froerr_opinf:%}",
            sep="\n",
        )

        projerr_in_time = opinf.post.lp_error(
            X_train,
            basis.project(X_train),
            normalize=True,
        )[1]

        plot_errors_over_time(
            [X_ROM],
            ["ROM Error"],
            exp.T,
            projerr_in_time,
            X_train,
            save_path=config.evaluations_path,
        )

        # %% output
        # Solve least square problem
        # || X*C.T - Y ||
        logging.info(f"Solve output least square problem..")
        CT = la.lstsq(X_train_comp.T, Y_train_fit.T, **rom.solver_.options)[0]

        Y_ROM = CT.T @ x_red

        plot_trajectory_over_time(
            exp.T,
            Y_train,
            Y_ROM,
            trajectory_type="output",
            save_path=config.evaluations_path,
            save_name=f"output_over_time_r{basis.shape[1]}",
        )

        n_u = (CT.T).shape[0]
        oi_abcd_fom = LTIModel.from_matrices(
            A=A_OI,
            B=rom.B_.entries,
            C=CT.T,
            D=np.zeros((n_u, n_u)),
        )

        # use sim for testing difference
        logging.info(f"Solve IMR separately...")
        u_red_imr, x_red_imr, y_red_imr = sim(oi_abcd_fom, exp.u, exp.T, x0_)
        X_ROM_IMR = basis.decompress(x_red_imr)
        plot_trajectory_over_time(
            exp.T,
            X_train,
            X_ROM_IMR,
            save_path=config.evaluations_path,
            save_name=f"state_over_time_r{basis.shape[1]}_imr",
        )
        plot_trajectory_over_time(
            exp.T,
            Y_train,
            y_red_imr,
            trajectory_type="output",
            save_path=config.evaluations_path,
            save_name=f"output_over_time_r{basis.shape[1]}_imr",
        )

        # %% create pH system
        logging.info(f"Create pH system from state-space system...")
        T, phlti_model = get_Riccati_transform(oi_abcd_fom, get_phlti=True)

        # %% Evaluation
        lti_dict["Original"] = exp.fom
        lti_dict["OI_PKG_ABCD"] = oi_abcd_fom
        if phlti_model is not None:
            lti_dict["OI_PKG_PH"] = phlti_model

        use_train_data = True
        if use_train_data:
            logging.warning(f"Training data used for results!")

        logging.info("Evaluate")
        evaluate(
            exp,
            lti_dict,
            compute_hinf=False,
            use_train_data=True,
            x0=x0_,
            # method="explicit_euler",
        )
        print("debug stop")


def plot_trajectory_over_time(
    time,
    X_orig,
    X_ROM=None,
    trajectory_type="state",
    save_path="",
    save_name="states_over_time",
):

    num_plots = X_orig.shape[0]
    if num_plots > 6:
        num_plots = 6
    idx_state = np.arange(0, num_plots)

    if num_plots == 1:
        # TODO: Change
        # Quick fix to circumvent axes not subscriptable error
        num_plots = 2

    if trajectory_type == "state":
        ylabel = "x"
    elif trajectory_type == "output":
        ylabel = "y"
    else:
        raise ValueError(f"Unknown option of trajectory_type: {trajectory_type}.")

    # plot
    fig, ax = plt.subplots(num_plots, 1, figsize=(5.701, 3.5), dpi=300, sharex="all")
    for i, i_idx in enumerate(idx_state):
        ax[i].plot(time, X_orig[i_idx, :], label=rf"$x_{{orig}}$")
        if X_ROM is not None:
            ax[i].plot(time, X_ROM[i_idx, :], linestyle="dashed", label=rf"$x_{{id}}$")
        ax[i].set_ylabel(
            rf"${ylabel}_{{{i}}}$",
            rotation="horizontal",
            ha="center",
            va="center",
        )
        ax[i].grid(linestyle=":", linewidth=1)
        i += 1
    plt.xlabel("time t [s]")
    fig.align_ylabels(ax[:])
    plt.legend(bbox_to_anchor=(1.04, 0.0), loc="lower left", borderaxespad=0.0)
    # fig.legend(loc='outside center right', bbox_to_anchor=(1.3, 0.6))
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig(os.path.join(save_path, f"{save_name}.png"))


def implicit_euler(t, q0, A, B, U):
    """Solve the system

        dq / dt = Aq(t) + Bu(t),    q(0) = q0,

    over a uniform time domain via the implicit Euler method.

    Parameters
    ----------
    t : (k,) ndarray
        Uniform time array over which to solve the ODE.
    q0 : (n,) ndarray
        Initial condition.
    A : (n, n) ndarray
        State matrix.
    B : (n,) or (n, 1) ndarray
        Input matrix.
    U : (k,) ndarray
        Inputs over the time array.

    Returns
    -------
    q : (n, k) ndarray
        Solution to the ODE at time t; that is, q[:,j] is the
        computed solution corresponding to time t[j].
    """
    # Check and store dimensions.
    k = len(t)
    n = len(q0)
    # B = np.ravel(B)
    assert A.shape == (n, n)
    assert B.shape[0] == n
    assert B.shape[1] == U.shape[0]
    assert U.shape[1] == k

    I = np.eye(n)

    # Check that the time step is uniform.
    dt = t[1] - t[0]
    assert np.allclose(np.diff(t), dt)

    # Factor I - dt*A for quick solving at each time step.
    factored = la.lu_factor(I - dt * A)

    # Solve the problem by stepping in time.
    q = np.empty((n, k))
    q[:, 0] = q0.copy()
    for j in range(1, k):
        q[:, j] = la.lu_solve(factored, q[:, j - 1] + dt * B @ U[:, j])

    return q


def plot_errors_over_time(
    Zlist, labels, t, projerr_in_time, Q_all, save_path="", save_name="error_over_time"
):
    """Plot normalized absolute projection error and ROM errors
    as a function of time.

    Parameters
    ----------
    Zlist : list((n, k) ndarrays)
        List of reduced-order model solutions.
    labels : list(str)
        Labels for each of the reduced-order models.
    """
    fig, ax = plt.subplots(1, 1)

    ax.semilogy(t, projerr_in_time, "C3", label="Projection Error")
    colors = ["C0", "C5"]
    for Z, label, c in zip(Zlist, labels, colors[: len(Zlist)]):
        rel_err = opinf.post.lp_error(Q_all, Z, normalize=True)[1]
        plt.semilogy(t, rel_err, c, label=label)

    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("Normalized absolute error")
    ax.legend(loc="lower right")
    plt.show()
    plt.savefig(os.path.join(save_path, f"{save_name}.png"))


if __name__ == "__main__":
    main()
    print("debug stop")
