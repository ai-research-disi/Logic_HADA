# -*- coding: utf-8 -*-
"""
Created on Thu Mar 2 16:37:45 2023

@author: EleMisi
"""

import const_define as cd
from utils.build_model_symbolic import build_and_solve_EML
from utils.util_functions import define_algs_dict, load_var_intervals, define_logic_rules

# Specify the ML model(s) and algorithm(s) of interest
# TODO: We focused on 'no_input-memory_DecisionTree_MaxDepth10' with ANTICIPATE algorithm.
#  Extend the procedure to the other models.
ML_MODELS = [cd.MODEL_DIR + '/no_input-memory_DecisionTree_MaxDepth10']
ALGS = ['ANTICIPATE']

if __name__ == '__main__':
    # Define dictionary for the desired ml model(s) and algorithm(s)
    algs_dict = define_algs_dict(ml_models=ML_MODELS, algs=ALGS)

    # Load variables intervals
    globmaxdict, globmindict = load_var_intervals(algs_dict=algs_dict)

    # Define logic rules extracted by GridREx
    logic_constraints = define_logic_rules(algs=ALGS, ml_models=ML_MODELS)

    # Define user constraints
    # TODO: usage examples are in Boscarini's code
    user_constraints = {}
    user_constraints['variable'] = []  # ['time(sec)']
    user_constraints['type'] = []  # ['<=']
    user_constraints['value'] = []  # [70, 90] min functioning values

    # Define objective function
    objective_var = 'sol(keuro)'
    objective_type = 'min'

    # Build and solve the problem
    build_and_solve_EML(df_mins=globmindict,
                        df_maxs=globmaxdict,
                        user_constraints=user_constraints,
                        logic_constraints=logic_constraints,
                        objective_type=objective_type,
                        objective_var=objective_var,
                        model_name="test_model",
                        enable_var_type=False,
                        inst_descr=cd.INSTANCE_FEATURES,
                        ml_trgt=cd.ML_TARGETS,
                        algs=algs_dict)
