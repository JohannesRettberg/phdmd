import logging
import os

import numpy as np

import pymor.core.logger

from tqdm import tqdm

from pymor.algorithms.to_matrix import to_matrix
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.reductors.basic import LTIPGReductor
from pymor.basic import project

from phdmd import config

from phdmd.data.generate import generate
from phdmd.data.generate import sim
from phdmd.evaluation.evaluation import h_norm
from phdmd.utils.plotting import plot

import cProfile
import pstats
from matplotlib import pyplot as plt
import scipy
from phdmd.utils.preprocess_data import get_Riccati_transform, get_initial_energy_matrix, perturb_initial_energy_matrix


def main():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    pymor.core.logger.set_log_levels({'pymor': 'WARNING'})

    if not os.path.exists(config.simulations_path):
        os.makedirs(config.simulations_path)

    if not os.path.exists(config.evaluations_path):
        os.makedirs(config.evaluations_path)

    exp = config.mimo_msd_exp
    logging.info(f'Experiment: {exp.name}')

    reduced_orders = np.array(range(2, 102, 10)) # np.array(range(2, 102, 2))
    noise_levels = np.array([None]) # np.array([None, 1e-4, 1e-6])

    num_methods = len(exp.methods) + 1

    h2_norms = np.zeros((len(reduced_orders), num_methods * len(noise_levels)))
    hinf_norms = np.zeros((len(reduced_orders), num_methods * len(noise_levels)))
    labels = np.empty(num_methods * len(noise_levels), dtype=object)
    
    use_energy_weighted_POD = False  #[True, False]
    experiment_name = f'{exp.name}_ePOD{use_energy_weighted_POD}_h_norms'

    error_list = []
    # for use_energy_weighted_POD in use_energy_weighted_PODs:
    for i, noise in enumerate(noise_levels):

        if noise is not None:
            experiment_name_i = experiment_name + f'_noise_{noise:.0e}'
        else:
            experiment_name_i = experiment_name

        if not os.path.exists(config.evaluations_path + '/' + f'{experiment_name_i}.npz'):

            n = exp.fom.order
            use_Berlin = exp.use_Berlin

            H, Q = get_initial_energy_matrix(exp)
            logging.info(f'State dimension n={n}')


            if exp.use_Riccatti_transform:
                T = get_Riccati_transform(exp)

            if exp.perturb_energy_matrix:
                H,Q = perturb_initial_energy_matrix(exp,H,Q) 

            # Set noise for the experiment
            exp.noise = noise
            # Generate/Load training data
            X_train, Y_train, U_train = generate(exp)
            # test data
            U_test, X_test, Y_test = sim(exp.fom, exp.u_test, exp.T_test, exp.x0_test, method=exp.time_stepper)

            if exp.use_Riccatti_transform:
                X_train = T@X_train

            h2_norms_i = np.zeros((len(reduced_orders), num_methods))
            hinf_norms_i = np.zeros((len(reduced_orders), num_methods))

            # POD
            
            if not os.path.exists(config.simulations_path + '/' + exp.name + '_POD.npz') or config.force_simulation:
                logging.info(f'Calculating POD from training data.')
                if use_energy_weighted_POD:
                    VV, S = energy_weighted_POD(X_train,H)
                else:
                    VV, S, _  = np.linalg.svd(X_train, full_matrices=False)
                np.savez(os.path.join(config.simulations_path, exp.name + '_POD.npz'),
                VV=VV, S=S)
            else:
                POD_npz = np.load(os.path.join(config.simulations_path, exp.name + '_POD.npz'))
                VV = POD_npz['VV']
                S = POD_npz['S']

            if use_energy_weighted_POD:
                X_train_projected_energy = VV@np.transpose(VV)@H@X_train
                proj_error_train_energy_fro = np.linalg.norm(X_train - X_train_projected_energy, ord='fro')
                logging.info(f"Frobenius norm of the energy projected error using all singular values: {proj_error_train_energy_fro}")
            else:
                X_train_projected = VV@np.transpose(VV)@X_train
                proj_error_train_fro = np.linalg.norm(X_train - X_train_projected, ord='fro')            
                logging.info(f"Frobenius norm of the projection error using all singular values: {proj_error_train_fro}")
                # proj_error_train_2 = np.linalg.norm(X_train - X_train_projected, ord=2)
            

            for j, r in enumerate(tqdm(reduced_orders)):
                lti_dict = {}

                V = NumpyVectorSpace.from_numpy(VV[:, :r].T, id='STATE')
                if use_energy_weighted_POD:
                    X_train_projected_r = np.transpose(V.to_numpy())@V.to_numpy()@H@X_train
                    # proj_error_train_2_r = np.linalg.norm(X_train - X_train_projected_r, ord=2)
                    proj_error_train_fro_r = np.linalg.norm(X_train - X_train_projected_r, ord='fro')
                    logging.info(f"Frobenius norm of the energy projected error using {r} singular values: {proj_error_train_fro_r}")

                else:
                    X_train_projected_r = np.transpose(V.to_numpy())@V.to_numpy()@X_train
                    # proj_error_train_2_r = np.linalg.norm(X_train - X_train_projected_r, ord=2)
                    proj_error_train_fro_r = np.linalg.norm(X_train - X_train_projected_r, ord='fro')
                    logging.info(f"Frobenius norm of the projection error using {r} singular values: {proj_error_train_fro_r}")

                # Transofrm data
                X_train_red = to_matrix(project(NumpyMatrixOperator(X_train, range_id='STATE'), V, None))
                if use_Berlin:
                    H_red = to_matrix(project(NumpyMatrixOperator(H, source_id='STATE', range_id='STATE'), V, V))
                    Q_red = None
                else:
                    Q_red = to_matrix(project(NumpyMatrixOperator(Q, source_id='STATE', range_id='STATE'), V, V))
                    H_red = None
                # Transform data
                if use_energy_weighted_POD:
                    pod_type = "ePOD"
                    VH = NumpyVectorSpace.from_numpy(VV[:, :r].T@H, id='STATE')
                    X_train_red = to_matrix(project(NumpyMatrixOperator(X_train, range_id='STATE'), VH, None))
                    # project initial condition
                    exp.x0_test_red = to_matrix(project(NumpyMatrixOperator(exp.x0_test[:,np.newaxis], range_id='STATE'), VH, None))
                    H_red = np.eye(r)

                    # POD with Q lhs
                    pg_reductor = LTIPGReductor(exp.fom.to_lti(), VH, V)
                    lti_pod = pg_reductor.reduce()
                    lti_dict[pod_type] = lti_pod
                else:      
                    pod_type = "POD"          
                    X_train_red = to_matrix(project(NumpyMatrixOperator(X_train, range_id='STATE'), V, None))
                    # project initial condition
                    exp.x0_test_red = to_matrix(project(NumpyMatrixOperator(exp.x0_test[:,np.newaxis], range_id='STATE'), V, None))
                    H_red = to_matrix(project(NumpyMatrixOperator(H, source_id='STATE', range_id='STATE'), V, V))

                    # POD with Q lhs
                    pg_reductor = LTIPGReductor(exp.fom.to_lti(), V, V)
                    lti_pod = pg_reductor.reduce()
                    lti_dict[pod_type] = lti_pod

                # Perform methods
                for method in exp.methods:
                    if method.name == 'OI' and use_energy_weighted_POD:
                        method.name = 'eOI'
                    elif method.name == 'pHDMD' and use_energy_weighted_POD:
                        method.name = 'epHDMD'
                    elif method.name == 'DMD' and use_energy_weighted_POD:
                        method.name = 'eDMD'
                        
                    lti = method(X_train_red, Y_train, U_train, exp.delta, use_Berlin=use_Berlin, H=H_red, Q=Q_red)
                    lti_dict[method.name] = lti

                    U_red, X_red, Y_red = sim(lti_dict[method.name], exp.u, exp.T, x0 = None, method=exp.time_stepper)
                    plt.figure()
                    plt.plot(np.transpose(Y_train),label="y")
                    plt.plot(np.transpose(Y_red),label="y_red")
                    plt.title(f"MSD model with train data for {method.name}, r={r}")
                    plt.legend()
                    plt.savefig(os.path.join(config.plots_path,f"train_y_r{r}_{pod_type}_{method.name}_noise{noise}.png"))
                    plt.close()

                    U_test_red, X_test_red, Y_test_red = sim(lti_dict[method.name], exp.u_test, exp.T_test, x0 = exp.x0_test_red, method=exp.time_stepper)
                    plt.figure()
                    plt.plot(np.transpose(Y_test),label="y")
                    plt.plot(np.transpose(Y_test_red),label="y_red")
                    plt.title(f"MSD model with test data for {method.name}, r={r}")
                    plt.legend()
                    plt.savefig((os.path.join(config.plots_path,f"test_y_r{r}_{pod_type}_{method.name}_noise{noise}.png")))  

                    X_test_red_rec = V.to_numpy().T@X_test_red
                    X_error_abs = np.abs(X_test - X_test_red_rec)
                    X_error_rel = np.linalg.norm(X_error_abs, axis=0) / np.linalg.norm(X_test, axis=0).mean()
                    # output error
                    Y_error_abs = np.abs(Y_test - Y_test_red)
                    Y_error_rel = Y_error_abs / Y_test.mean()
                    error_dict = {
                        "X_error_abs": X_error_abs,
                        "X_error_rel": X_error_rel,
                        "Y_error_abs": Y_error_abs,
                        "Y_error_rel": Y_error_rel,
                        "method": method.name,
                        "r": r,
                        "noise": noise,
                        "use_energy_weighted_POD": use_energy_weighted_POD,
                    }
                    error_list.append(error_dict)
                    plt.figure()
                    plt.plot(np.transpose(X_error_rel),label="e_rel")
                    plt.title(f"Guitar model, rel. state error 2-norm, r={r}")
                    plt.legend()
                    plt.savefig((os.path.join(config.plots_path,f"error_rel_r{r}_{pod_type}_{method.name}_noise{noise}.png"))) 

                h2, hinf = h_norm(exp.fom, list(lti_dict.values()), list(lti_dict.keys()), compute_hinf=True)
                h2_norms_i[j] = h2
                hinf_norms_i[j] = hinf

            labels_i = np.array(list(lti_dict.keys()))
            np.savez(os.path.join(config.evaluations_path, f'{experiment_name_i}_{pod_type}'), h2_norms=h2_norms_i,
                    hinf_norms=hinf_norms_i, labels=labels_i, reduced_orders=reduced_orders)
        else:
            npzfile = np.load(os.path.join(config.evaluations_path, f'{experiment_name_i}_{pod_type}.npz'))
            h2_norms_i = npzfile['h2_norms']
            hinf_norms_i = npzfile['hinf_norms']
            labels_i = npzfile['labels']
            reduced_orders = npzfile['reduced_orders']

        if noise is not None:
            labels_i = np.core.defchararray.add(labels_i, f' ($s={noise_levels[i]:.0e}$)')

        labels[i * num_methods:(i + 1) * num_methods] = labels_i
        h2_norms[:, i * num_methods:(i + 1) * num_methods] = h2_norms_i
        hinf_norms[:, i * num_methods:(i + 1) * num_methods] = hinf_norms_i


    # save error
    np.savez(os.path.join(config.evaluations_path, f'{experiment_name_i}_{pod_type}_error.npz'), error_list=error_list)

    c = config.colors[:3]
    c = np.tile(c, (len(noise_levels), 1))

    ls = np.array(['-', '-', '-', '--', '--', '--', ':', ':', ':'])

    markers = np.array(['o', 's', 'D'])
    markers = np.tile(markers, len(noise_levels))

    markevery = 10

    if exp.perturb_energy_matrix:
        if isinstance(exp.perturb_value,str):
            add_perturb_name = f"_perturb_{exp.perturb_value}"
        else:
            add_perturb_name = f"_perturb{exp.perturb_value:.0e}"
    else:
        add_perturb_name = ""

    if exp.use_Riccatti_transform:
        add_Ricc_name = f"_Ricc{exp.use_Riccatti_transform}"
    else:
        add_Ricc_name = ""

    add_plot_name = f"_Bform{exp.use_Berlin}_init_{exp.HQ_init_strat}{add_perturb_name}{add_Ricc_name}"

    plot(reduced_orders, h2_norms.T[:, np.newaxis, :], label=labels,
         c=c, ls=ls, marker=markers, markevery=markevery,
         yscale='log', ylabel='$\mathcal{H}_2$ error', grid=True, subplots=False,
         xlabel='Reduced order', fraction=1, name=f'{experiment_name}_h2_{add_plot_name}')


def profile_run(code_name_string='main()'):
    prof = cProfile.Profile()
    prof.run(code_name_string)
    prof.dump_stats(f"profile_output_{code_name_string.replace('(','').replace(')','')}.prof")

    stream = open(f"profile_output_{code_name_string.replace('(','').replace(')','')}.txt", 'w')
    stats = pstats.Stats(f"profile_output_{code_name_string.replace('(','').replace(')','')}.prof", stream=stream)
    stats.sort_stats('cumtime')
    stats.print_stats()


def energy_weighted_POD(X_train,H):
    if X_train.shape[0] < X_train.shape[1]:
        # eigenvalue decomposition of size nxn
        # weight with energy matrix
        X_train_weighted = H@(X_train@np.transpose(X_train))
        sing_vals, sing_vecs = np.linalg.eig(X_train_weighted)
        # sort in descending order
        sort_idx = np.argsort(sing_vals)[::-1]
        sing_vecs = np.linalg.solve(H,sing_vecs[:,sort_idx])
        # norm the basis with respect to H scalar product
        weighting_values = np.transpose(sing_vecs)@H@sing_vecs
        weighting_values = np.diag(1/np.sqrt(np.diag(weighting_values)))
        sing_vecs = sing_vecs@weighting_values
        # sort singular values in descending order
        sing_vals = sing_vals[sort_idx]
        print(sing_vecs)

    else:
        # eigenvalue decomposition of size n_txn_t
        # weight energy matrix
        X_train_weighted = np.transpose(X_train)@H@X_train
        # sing_vals, sing_vecs = np.linalg.eig(X_train_weighted)
        # solve eigenvalue problem of symmetric matrix
        sing_vals, sing_vecs = scipy.linalg.eigh(X_train_weighted)
        # sort in descending order
        sort_idx = np.argsort(sing_vals)[::-1]
        sing_vals = sing_vals[sort_idx]
        sing_vecs = sing_vecs[:,sort_idx]
        # convert to left-hand singular values
        sing_vals[sing_vals < 1e-10] = np.inf # mask small singular values as invalid (prevent divide by zero error)
        sing_vals = np.ma.masked_invalid(sing_vals)
        sing_vecs = X_train@sing_vecs@np.diag(1/np.sqrt(sing_vals))
        print(sing_vecs)

    # bool_list = []
    # for i in range(X_train.shape[1]):
    #     if np.transpose(X_train[:,i])@H@X_train[:,i] >= 0:
    #         bool_list.append(True)
    #     else:
    #         bool_list.append(False)

    if np.iscomplex(sing_vecs).any():
        # take only real part of basis matrix
        sing_vecs = np.real(sing_vecs)
    VV = sing_vecs
    S = sing_vals

    return VV, S


if __name__ == "__main__":
    profile_run('main()')
    print('debug stop')
