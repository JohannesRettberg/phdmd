import os

from phdmd.utils.plotting import plot
from phdmd import config
import numpy as np

plot_name = f"h2_norm_PHDMD_comp"

exp = config.mimo_msd_exp
reduced_orders = np.array(range(2, 102, 4))  # default: range(2, 102, 2)
noise_levels = np.array([None, 1e-4, 1e-6])  # default: [None, 1e-4, 1e-6]
num_methods = len(exp.methods) + 1
use_energy_weighted_PODs = [True, False]  # default: True (ePOD) or False (POD)

# method index from [POD, OI, PHDMD], e.g. [0,1,2] for all methods
num_all_methods = 3
method_idx = np.array([2])

# h2_norms_all = np.zeros((len(reduced_orders),len(noise_levels)*num_methods*len(use_energy_weighted_PODs)))
iteration_num = 0
for i_pod, use_energy_weighted_POD in enumerate(use_energy_weighted_PODs):
    if use_energy_weighted_POD:
        pod_type = "ePOD"
    else:
        pod_type = "POD"

    for i_noise, noise in enumerate(noise_levels):
        experiment_name = f"{exp.name}_ePOD{use_energy_weighted_POD}_h_norms"

        if noise is not None:
            experiment_name_i = experiment_name + f"_noise_{noise:.0e}"
        else:
            experiment_name_i = experiment_name

        npzdata = np.load(
            os.path.join(config.evaluations_path, f"{experiment_name_i}_{pod_type}.npz")
        )
        h2_norms = npzdata["h2_norms"]
        hinf_norms = npzdata["hinf_norms"]
        labels = npzdata["labels"]
        reduced_orders = npzdata["reduced_orders"]

        if noise is not None:
            labels = np.core.defchararray.add(
                labels, f" ($s={noise_levels[i_noise]:.0e}$)"
            )

        if iteration_num == 0:
            h2_norms_all = h2_norms[:, method_idx]
            labels_all = labels[method_idx]
        else:
            h2_norms_all = np.concatenate(
                (h2_norms_all, h2_norms[:, method_idx]), axis=1
            )
            labels_all = np.concatenate((labels_all, labels[method_idx]), axis=0)
        iteration_num += 1

    if i_pod == 0:
        c = config.colors[method_idx + use_energy_weighted_POD * num_all_methods]
        c_all = np.tile(c, (len(noise_levels), 1))
    else:
        c = config.colors[method_idx + use_energy_weighted_POD * num_all_methods]
        c = np.tile(c, (len(noise_levels), 1))
        c_all = np.concatenate((c_all, c), axis=0)


ls = np.array(["-", "-", "-", "--", "--", "--", ":", ":", ":"])

markers = np.array(["o", "s", "D"])
markers = np.tile(markers, len(noise_levels))

markevery = 10
plot(
    reduced_orders,
    h2_norms_all.T[:, np.newaxis, :],
    label=labels_all,
    c=c_all,
    ls=ls,
    marker=markers,
    markevery=markevery,
    yscale="log",
    ylabel="$\mathcal{H}_2$ error",
    grid=True,
    subplots=False,
    xlabel="Reduced order",
    fraction=1,
    name=plot_name,
)
