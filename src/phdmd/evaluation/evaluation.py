import logging

import numpy as np

from phdmd import config
from phdmd.data.generate import sim
from phdmd.utils.plotting import plot


def h_norm(lti_ref, lti_approx, label, compute_hinf=True):
    """
    Compute the H2 and Hinf norm of the error between the reference and the approximated system.

    Parameters
    ----------
    lti_ref : pymor.models.iosys.LTIModel
        The reference system.
    lti_approx : pymor.models.iosys.LTIModel
        The approximated system.
    label : str
        Label for the approximated system.
    compute_hinf : bool, optional
        If True, the Hinf norm is computed. Default True.

    Returns
    -------
    h2_norms : numpy.ndarray
        The H2 norm of the error.
    hinf_norms : numpy.ndarray
        The Hinf norm of the error.
    """
    if not isinstance(lti_approx, list):
        lti_approx = [lti_approx]

    if not isinstance(label, list):
        label = [label]

    logging.info("H norm errors:")
    h2_ref = lti_ref.h2_norm()

    if compute_hinf:
        hinf_ref = lti_ref.hinf_norm()

    h2_norms = np.zeros(len(lti_approx))
    hinf_norms = np.zeros(len(lti_approx))

    for i, lti_appr in enumerate(lti_approx):
        try:
            lti_error = lti_ref - lti_appr
        except AssertionError:
            lti_error = lti_ref - lti_appr.to_continuous()
        logging.info(label[i])

        h2 = lti_error.h2_norm()
        h2 = h2 / h2_ref
        logging.info(f"Relative H2 error: {h2:.2e}")
        h2_norms[i] = h2

        if compute_hinf:
            hinf = lti_error.hinf_norm()
            hinf = hinf / hinf_ref
            logging.info(f"Relative Hinf error: {hinf:.2e}")
            hinf_norms[i] = hinf

    return h2_norms, hinf_norms


def evaluate(
    exp,
    lti_dict,
    compute_hinf=True,
    x0=None,
    method="implicit_midpoint",
    use_train_data=False,
):
    """
    Evaluate the experiment for given approximated systems.
    Generates plots and computes the H2 and (optional) Hinf norm of the error.

    Parameters
    ----------
    exp : Experiment
        The experiment to evaluate.
    lti_dict : dict
        Dictionary of approximated systems.
    """
    Y_list = []
    Y_error_list = []
    lti_error_list = []
    labels = []

    lti_list = list(lti_dict.values())
    labels_i = list(lti_dict.keys())
    labels += labels_i
    for j in range(len(lti_list)):
        logging.info(f"Evaluate {labels_i[j]}")
        # initial condition (for reduced version)
        r = None  # reduced dimension
        if x0 is None:
            if use_train_data:
                x0_ = exp.x0
            else:
                x0_ = exp.x0_test
        else:
            if use_train_data:
                if lti_list[j].order == exp.x0.shape[0]:
                    # FOM reference system
                    x0_ = exp.x0
                else:
                    r = lti_list[j].order
                    x0_ = x0
            else:
                if lti_list[j].order == exp.x0_test.shape[0]:
                    # FOM reference system
                    x0_ = exp.x0_test
                else:
                    r = lti_list[j].order
                    x0_ = x0

        # Simulate for testing input
        if use_train_data:
            U, X, Y = sim(lti_list[j], exp.u, exp.T, x0_, method=method)
            time_data = exp.T
            data_result_naming = "training"
        else:
            U, X, Y = sim(lti_list[j], exp.u_test, exp.T_test, x0_, method=method)
            time_data = exp.T_test
            data_result_naming = "testing"
        Y_list.append(Y)

        if j > 0:
            Y_error = np.abs(Y_list[0] - Y_list[j])
            Y_error_list.append(Y_error)
            try:
                lti_error = lti_list[0] - lti_list[j]
            except AssertionError:
                lti_error = lti_list[0] - lti_list[j].to_continuous()

            lti_error_list.append(lti_error)

    repeat_time_values = int(X.shape[1] / time_data.shape[0])
    # Trajectories
    # plot(exp.T_test, U, label='$u$', ylabel='Input', xlabel='Time (s)', legend='upper right',
    #      fraction=config.fraction, name=exp.name + '_testing_input')

    ls = len(labels) * ["--"]
    ls[0] = "-"
    testing_output_labels = [
        "$y$",
        *[r"$\widetilde{y}_{\mathrm{" + l + "}}$" for l in labels[1:]],
    ]

    if exp.perturb_energy_matrix:
        if isinstance(exp.perturb_value, str):
            add_perturb_name = f"_perturb_{exp.perturb_value}"
        else:
            add_perturb_name = f"_perturb{exp.perturb_value:.0e}"
    else:
        add_perturb_name = ""

    if exp.use_Riccatti_transform:
        add_Ricc_name = f"_Ricc{exp.use_Riccatti_transform}"
    else:
        add_Ricc_name = ""

    if exp.use_cholesky_like_fact:
        add_chol_fac = f"_Chol{exp.use_cholesky_like_fact}"
    else:
        add_chol_fac = ""

    if exp.use_cvx:
        add_cvx = f"_cvx{exp.use_cvx}_Jknown{exp.use_known_J}"
    else:
        add_cvx = ""

    if exp.use_projection_of_A:
        add_proj_A_snd = f"_projAsnd{exp.use_projection_of_A}"
    else:
        add_proj_A_snd = ""

    if r is not None:
        add_red_dim = f"_r{r}"
    else:
        add_red_dim = ""

    method_name = f"{labels_i[j]}"
    add_plot_name = f"_{method_name}_Bform{exp.use_Berlin}_init_{exp.HQ_init_strat}{add_perturb_name}{add_Ricc_name}{add_chol_fac}{add_cvx}{add_proj_A_snd}{add_red_dim}"

    plot(
        time_data,
        Y_list,
        label=testing_output_labels,
        c=config.colors,
        xlabel="Time",
        # ylim=[-1.5, 0.5],
        ylabel=f"{data_result_naming} output",
        name=f"{exp.name}_{data_result_naming}_output{add_plot_name}",
        ls=ls,
        fraction=config.fraction,
    )

    # Absolute Error of the trajectories
    plot(
        np.tile(time_data, (repeat_time_values, 1)).ravel(),
        Y_error_list,
        label=labels[1:],
        c=config.colors[1:],
        yscale="log",
        xlabel="Time",
        ylabel="Absolute error",
        name=f"{exp.name}_abs_error{add_plot_name}",
        fraction=config.fraction,
    )

    # H-norm calculation fails for the poro benchmark system
    # if exp.model != "poro":
    # h_norm(lti_list[0], lti_list[1:], labels[1:], compute_hinf=compute_hinf)


class Evaluation:
    def __init__(self) -> None:
        pass
