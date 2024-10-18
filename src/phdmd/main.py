import logging
import os
import numpy as np

from phdmd import config
from phdmd.data.generate import generate
from phdmd.evaluation.evaluation import evaluate
from pymor.algorithms.to_matrix import to_matrix
from phdmd.linalg.definiteness import project_spsd, project_spd
from phdmd.utils.preprocess_data import (
    get_Riccati_transform,
    get_initial_energy_matrix,
    perturb_initial_energy_matrix,
)
from phdmd.utils.postprocess_data import compare_matrices
from phdmd.algorithm.methods import CVXABCDPRMethod, PHDMDHMethod
from phdmd.data.data import Data

import cProfile
import pstats
from pymor.models.iosys import PHLTIModel, LTIModel
from phdmd.evaluation.result import Result


def main():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    if not os.path.exists(config.simulations_path):
        os.makedirs(config.simulations_path)

    if not os.path.exists(config.evaluations_path):
        os.makedirs(config.evaluations_path)

    # specify experiments in config.py
    # experiments = config.experiments
    experiments = [config.siso_msd_exp]

    for exp in experiments:
        logging.info(f"Experiment: {exp.name}")
        lti_dict = {}

        data_train = Data.from_experiment(exp)
        data_test = Data.from_fom(
            exp.fom, exp.u_test, exp.T_test, exp.x0_test, method=exp.time_stepper
        )

        if exp.use_Riccatti_transform:
            # T = get_Riccati_transform(exp.fom)
            data_train.get_Riccati_transform()
            data_test.get_Riccati_transform()

        n = exp.fom.order
        use_Berlin = exp.use_Berlin

        # H, Q = get_initial_energy_matrix(exp)

        # if exp.perturb_energy_matrix:
        #     H, Q = perturb_initial_energy_matrix(exp, H, Q)
        X_train, Y_train, U_train = data_train.data

        additional_data_init = {}
        if exp.HQ_init_strat == "Ham":
            additional_data_init["X"] = X_train[:, :]
            additional_data_init["project"] = False

        data_train.get_initial_energy_matrix(exp, additional_data=additional_data_init)
        if exp.perturb_energy_matrix:
            data_train.perturb_initial_energy_matrix(exp)
        H, Q = data_train.initial_energy_matrix

        if exp.use_known_J:
            assert exp.use_cvx
            J_known = exp.J
        else:
            J_known = None

        logging.info(f"State dimension n = {n}")
        logging.info(f"Step size delta = {exp.delta:.2e}")
        lti_dict["Original"] = exp.fom

        # Generate/Load training data
        # X_train, Y_train, U_train = generate(exp)

        # if exp.use_Riccatti_transform:
        #     X_train = T @ X_train

        # Plot training input, analogously output or state data
        # plot(exp.T, U_train, label='$u$', ylabel='Input', xlabel='Time (s)',
        #      fraction=config.fraction, name=exp.name + '_training_input')

        # Perform methods
        for method in exp.methods:
            if isinstance(method, CVXABCDPRMethod):
                add_method_inputs = {
                    "gillis_options": exp.gillis_options,
                    "constraint_type": exp.constraint_type,
                }
            elif isinstance(method, PHDMDHMethod):
                add_method_inputs = {"ordering": exp.ordering}
            else:
                add_method_inputs = {}

            lti = method(
                X_train,
                Y_train,
                U_train,
                exp.delta,
                use_Berlin=use_Berlin,
                H=H,
                Q=Q,
                use_cvx=exp.use_cvx,
                J_known=J_known,
                **add_method_inputs,
            )
            lti_dict[method.name] = lti

        # get results
        results = Result(
            lti_dict, exp, data_train, data_test, save_path=config.plots_path
        )
        results.get_all_results()

        # filename = "test_save"

        # results_saved = results.save(filename)
        # results_loaded = results.restore_static(results_saved[0],results_saved[1])

        # with open(f"{filename}.txt", "wb") as file_:
        #     pickle.dump([results],file_)
        # results.save_instance(filename,results)
        # results_loaded = Result.load_instance(filename)

        latex_output = True
        results.compare_matrices(lti, latex_output=latex_output)

        # Evaluation
        # logging.info("Evaluate")
        # evaluate(exp, lti_dict, compute_hinf=False)


def profile_run(code_name_string="main()"):
    prof = cProfile.Profile()
    prof.run(code_name_string)
    # prof.sort_stats("cumtime")
    prof.dump_stats(f"profile_output_{code_name_string}.prof")

    stream = open(f"profile_output_{code_name_string}.txt", "w")
    stats = pstats.Stats(f"profile_output_{code_name_string}.prof", stream=stream)
    stats.sort_stats("cumtime")
    stats.print_stats()


if __name__ == "__main__":
    profile_run("main()")
    print("debug stop")
