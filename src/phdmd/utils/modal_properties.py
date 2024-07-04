import numpy as np
import os
import logging
from matplotlib import pyplot as plt


def compare_eigenmodes(lti1, lti2, save_name="", title=""):
    """
    compares eigenmodes of two systems lti1 and lti2 using MAC
    """
    logging.info(f"Calculating the eigenmodes.")

    _, eig_modes1 = calc_eigenvalues(lti1, modes=True)
    _, eig_modes2 = calc_eigenvalues(lti2, modes=True)

    modal_assurance_criterion(
        eig_modes1,
        eig_modes2,
        save_name=save_name,
        title=title,
    )


def modal_assurance_criterion(
    phi1,
    phi2=None,
    phi1_name="",
    phi2_name="",
    save_name="",
    title="",
):
    """
    Function that calculates the modal assurance criterion (MAC)
    and plots it.

    Parameters
    ----------
    phi1: (n,n_modes) np.array
        array with mode shapes
    phi2: (n,n_modes) np.array
        array with mode shapes

    """
    assert phi1.shape[0] >= phi1.shape[1]
    if phi2 is None:
        # compare mode shapes of one system
        phi2 = phi1
        phi2_name = phi1_name
    assert phi2.shape[0] >= phi2.shape[1]

    n_modes1 = phi1.shape[1]
    n_modes2 = phi2.shape[1]

    mac_values = np.zeros((n_modes1, n_modes2))
    for i in range(n_modes1):
        for j in range(n_modes2):
            mac_values[i, j] = mac(phi1[:, i], phi2[:, j])

    # plot mac values
    fig, ax = plt.subplots()
    im = ax.pcolormesh(mac_values)
    if title == "":
        title = f"Modal assurance criterion"
    plt.title(f"{title}")
    plt.xlabel(f"Mode {phi1_name}")
    plt.ylabel(f"Mode {phi2_name}")
    plt.xticks(np.arange(n_modes1), np.arange(n_modes1) + 1)
    plt.yticks(np.arange(n_modes2), np.arange(n_modes2) + 1)
    fig.colorbar(im, ax=ax)
    if save_name == "":
        save_name = "MAC_plot.png"
    else:
        save_name = save_name
    if not save_name.endswith(".png"):
        save_name = f"{save_name}.png"
    plt.savefig(f"{save_name}")


def mac(phi1, phi2):
    """This function calculates mac between phi1 and phi2
    Parameters
    ----------
    phi1: (n,) np.array
        vector with single mode shape
    phi2: (n,) np.array
        vector with single mode shape
    """
    return np.abs(phi1.conj().T @ phi2) ** 2 / (
        (phi1.conj().T @ phi1) * (phi2.conj().T @ phi2)
    )


def eigenvalue_comparison_from_lti_dicts(lti_dicts, max_system_size, save_name=""):
    """ """

    n_r = len(lti_dicts)  # number of reduced sizes
    n_methods = len(lti_dicts[0].keys())
    eig_vals_all = np.empty((max_system_size, n_r, n_methods)) * np.nan
    system_sizes = []
    for i_red, lti_dict in enumerate(lti_dicts):
        for i_method, (lti_name, lti_model) in enumerate(lti_dict.items()):
            if i_method == 0:
                # first method must be intrusive
                assert lti_name in ["POD", "ePOD"]
                system_size = lti_model.order
                system_sizes.append(system_size)
            eig_vals_all[: lti_model.order, i_red, i_method] = calc_eigenvalues(
                lti_model
            )

    # plot eigenvalues over reduced order
    if max_system_size > 6:
        # maximal number of 6 subplots
        number_plotted_eigenvalues = 6
    else:
        number_plotted_eigenvalues = max_system_size

    # save name
    if save_name == "":
        save_name = "eigenvalue_over_r.png"
    else:
        save_name = save_name
    if save_name.endswith(".png"):
        # save_name = f"{save_name}.png"
        save_name = os.path.splitext(save_name)[0]

    method_names = lti_dicts[0].keys()

    plt.rcParams["text.usetex"] = False  # leads to dvipng error
    cycler = plt.cycler(linestyle=["-", "--", "-.", ":"], color="bgrc")
    # plt.rcParams['backend'] = "pdf"
    plt.figure()
    fig, axs = plt.subplots(number_plotted_eigenvalues, 1)
    for i in range(number_plotted_eigenvalues):
        axs[i].set_prop_cycle(cycler)
        axs[i].plot(system_sizes, eig_vals_all[i, :, :])
        axs[i].set_ylabel(rf"\lambda_{i}")
    plt.show(block=False)
    plt.xlabel(f"red. dimension")
    plt.legend(method_names)
    plt.savefig(f"{save_name}.pdf")

    # save eigenvalues to .npz
    # save_data = os.path.splitext(save_name)[0]
    save_data = save_name
    np.savez(f"{save_data}.npz", eig_vals_all=eig_vals_all)


def calc_eigenvalues(lti_model, modes=False):
    """
    Calculate eigenvalues from lti_model
    modes (bool): define if eigenmodes shall be calculated
    """
    A, B, C, D, E = lti_model.to_abcde_matrices()
    # bring E to rhs if not identity
    if E is not None:
        if not (np.allclose(E, np.eye(E.shape[0]))):
            A = np.linalg.solve(E, A)

    if modes:
        eig_vals, eig_modes = np.linalg.eig(A)
        return eig_vals, eig_modes
    else:
        eig_vals = np.linalg.eigvals(A)
        return eig_vals
