import logging
import os
import numpy as np

from phdmd import config
from phdmd.data.generate import generate
from phdmd.evaluation.evaluation import evaluate
from phdmd.linalg.definiteness import project_spsd, project_spd
from phdmd.utils.preprocess_data import get_Riccati_transform, get_initial_energy_matrix, perturb_initial_energy_matrix

import cProfile
import pstats
from pymor.models.iosys import PHLTIModel, LTIModel

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
        logging.info(f'Experiment: {exp.name}')
        lti_dict = {}

        if exp.use_Riccatti_transform:
            T = get_Riccati_transform(exp)

        n = exp.fom.order
        use_Berlin = exp.use_Berlin

        H, Q = get_initial_energy_matrix(exp)

        if exp.perturb_energy_matrix:
            H,Q = perturb_initial_energy_matrix(exp,H,Q)             

        logging.info(f'State dimension n = {n}')
        logging.info(f'Step size delta = {exp.delta:.2e}')
        lti_dict['Original'] = exp.fom

        # Generate/Load training data
        X_train, Y_train, U_train = generate(exp)


        if exp.use_Riccatti_transform:
            X_train = T@X_train

        # Plot training input, analogously output or state data
        # plot(exp.T, U_train, label='$u$', ylabel='Input', xlabel='Time (s)',
        #      fraction=config.fraction, name=exp.name + '_training_input')

        # Perform methods
        for method in exp.methods:
            lti = method(X_train, Y_train, U_train, exp.delta, use_Berlin, H=H, Q=Q)
            lti_dict[method.name] = lti

        # Evaluation
        logging.info('Evaluate')
        evaluate(exp, lti_dict, compute_hinf=False)

def profile_run(code_name_string='main()'):
    prof = cProfile.Profile()
    prof.run(code_name_string)
    prof.sort_stats('cumtime')
    prof.dump_stats(f'profile_output_{code_name_string}.prof')

    stream = open(f'profile_output_{code_name_string}.txt', 'w')
    stats = pstats.Stats(f'profile_output_{code_name_string}.prof', stream=stream)
    stats.sort_stats('cumtime')
    stats.print_stats()


if __name__ == "__main__":
    profile_run('main()')
    print('debug stop')
