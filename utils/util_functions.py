# -*- coding: utf-8 -*-
"""
Created on Thu Mar 2 16:37:45 2023

@author: EleMisi
"""

import csv
import os
import time

import numpy as np
import pandas as pd

import const_define as cd


def define_algs_dict(ml_models: list, algs: list):
    """
    Returns the data structure defined by Boscarini.
    TODO: 1) adjust the data structure for the new HADA code.

    :param ml_models: list of ml models
    :param algs: list of algorithms (i.e., 'ANTICIPATE' and/or 'CONTINGENCY')

    :return: dict with the information of the desired algorithm(s) and ml model(s)
    """

    algs_dict = {}

    for alg in algs:
        algs_dict[alg] = {}
        algs_dict[alg]['dataset_cols'] = [cd.ALG_PARAM[alg]]
        algs_dict[alg]['alg_params'] = {'0': {'name': cd.ALG_PARAM[alg], 'type': int}}
        algs_dict[alg]['ml_model'] = {}
        for model in ml_models:
            algs_dict[alg]['ml_model'][model] = {}
            algs_dict[alg]['ml_model'][model]['target'] = cd.TARGET[alg][model]
            algs_dict[alg]['ml_model'][model]['features'] = cd.FEATURES[alg][model]

    return algs_dict


def load_var_intervals(algs_dict):
    """
    Wrap-up function of Boscarini's code.

    :param algs_dict: dict with the information of the algorithma and ml models.

    :return: dict with lower and upper bounds of the problem variables.
    """
    globminlist = []
    globmaxlist = []
    globmaxdict = {}
    globmindict = {}

    for alg in algs_dict.keys():
        """#### Load dataset"""
        path = os.path.join(cd.DATA_DIR, '{}_trainDataset.csv'.format(alg))
        df = pd.read_csv(path)

        # Removes header entries
        df = df[df['sol(keuro)'] != 'sol(keuro)']

        # Algorithm specific vars
        for var in algs_dict[alg]['alg_params'].values():
            df[var['name']] = df[var['name']].astype(var['type'])

        # Fixed stuff which is always there
        df['PV(kW)'] = df['PV(kW)'].map(lambda entry: entry[1:-1].split())
        df['PV(kW)'] = df['PV(kW)'].map(lambda entry: list(np.float_(entry)))
        df['Load(kW)'] = df['Load(kW)'].map(lambda entry: entry[1:-1].split())
        df['Load(kW)'] = df['Load(kW)'].map(lambda entry: list(np.float_(entry)))
        df['sol(keuro)'] = df['sol(keuro)'].astype(float)
        df['time(sec)'] = df['time(sec)'].astype(float)
        df['memAvg(MB)'] = df['memAvg(MB)'].astype(float)

        df['PV_mean'] = df['PV(kW)'].map(lambda entry: np.array(entry).mean())
        df['PV_std'] = df['PV(kW)'].map(lambda entry: np.array(entry).std())
        df['Load_mean'] = df['Load(kW)'].map(lambda entry: np.array(entry).mean())
        df['Load_std'] = df['Load(kW)'].map(lambda entry: np.array(entry).std())

        cur_cols = algs_dict[alg]['dataset_cols']
        cur_cols.extend(cd.INSTANCE_FEATURES + cd.ML_TARGETS)
        df_s = df[cur_cols]

        globminlist.append(df_s.min())
        globmaxlist.append(df_s.max())

        globmaxdict[alg] = df_s.max()
        globmindict[alg] = df_s.min()

    globmax = pd.DataFrame(globmaxlist).max()
    globmin = pd.DataFrame(globminlist).min()

    globmaxdict['glob'] = globmax
    globmindict['glob'] = globmin

    return globmaxdict, globmindict


def define_logic_rules(algs, ml_models):
    """
    Returns the logic rules for the desired algorithm(s) and model(s).
    TODO: this is a temporary function, needs to be improved and extended to consider all the algorithms and all the models.
    TODO: create a pickle file with the rules in the xlsx fil already transformed in the required dict format.

    :param algs: list of algorithm(s) of interest
    :param ml_models: list of ml model(s) of interest

    :return: dict containing the logic rules for the desired algorithm(s) and model(s)
    """

    # TODO: I focused on 'no_input-memory_DecisionTree_MaxDepth10' with ANTICIPATE algorithm.
    #  Extend the procedure to the other models.
    assert algs == ['ANTICIPATE'] and ml_models == [
        cd.MODEL_DIR + '/no_input-memory_DecisionTree_MaxDepth10'], "Currently supported algorithm and ml model are 'ANTICIPATE' and 'no_input-memory_DecisionTree_MaxDepth10' \n TODO: Extend to others "

    logic_constraints = {'ANTICIPATE': {}}
    logic_constraints['ANTICIPATE'][cd.MODEL_DIR + '/no_input-memory_DecisionTree_MaxDepth10'] = []
    logic_constraints['ANTICIPATE'][cd.MODEL_DIR + '/no_input-memory_DecisionTree_MaxDepth10'].append({
        'if': {
            'variable': ['y_nScenarios'],
            'type': ['range'],
            'value': [(0.99, 20.79)]
        },
        'then': {
            'variable': ['y_ANTICIPATE_memAvg(MB)'],
            'type': ['=='],
            'value': ['66.20 + 4.02 * y_nScenarios']
        }
    })

    logic_constraints['ANTICIPATE'][cd.MODEL_DIR + '/no_input-memory_DecisionTree_MaxDepth10'].append({
        'if': {
            'variable': ['y_nScenarios'],
            'type': ['range'],
            'value': [(20.79, 40.6)]
        },
        'then': {
            'variable': ['y_ANTICIPATE_memAvg(MB)'],
            'type': ['=='],
            'value': ['88.71 + 2.59 * y_nScenarios']
        }
    })

    logic_constraints['ANTICIPATE'][cd.MODEL_DIR + '/no_input-memory_DecisionTree_MaxDepth10'].append({
        'if': {
            'variable': ['y_nScenarios'],
            'type': ['range'],
            'value': [(40.6, 60.4)]
        },
        'then': {
            'variable': ['y_ANTICIPATE_memAvg(MB)'],
            'type': ['=='],
            'value': ['83.04 + 2.68 * y_nScenarios']
        }
    })

    logic_constraints['ANTICIPATE'][cd.MODEL_DIR + '/no_input-memory_DecisionTree_MaxDepth10'].append({
        'if':
            {'variable': ['y_nScenarios'],
             'type': ['range'],
             'value': [(60.4, 80.20)]}
        ,
        'then': {
            'variable': ['y_ANTICIPATE_memAvg(MB)'],
            'type': ['=='],
            'value': ['77.25 + 2.73 * y_nScenarios']
        }
    })

    logic_constraints['ANTICIPATE'][cd.MODEL_DIR + '/no_input-memory_DecisionTree_MaxDepth10'].append({
        'if':
            {'variable': ['y_nScenarios'],
             'type': ['range'],
             'value': [(80.20, 100.00)]}
        ,
        'then': {
            'variable': ['y_ANTICIPATE_memAvg(MB)'],
            'type': ['=='],
            'value': ['107.64 + 2.30 * y_nScenarios']
        }
    })

    return logic_constraints


def get_linear_expression(s: str):
    """
    Returns the input linear expression with the link to the cplex model variables.

    :param s: string containing the linear expression

    :return:
        A string with where the variables' name in the original linear expression have been replaced by the corresponding
        cplex model variables

    """
    l = s.split()
    for i, token in enumerate(l):
        try:
            eval(token)
            l[i] = token
        except Exception:
            if token in ['+', '-', '*']:
                l[i] = token
            else:
                var = f'mdl.get_var_by_name("{token}")'
                l[i] = var
    linear_expr = ' '.join(l)

    return linear_expr


def write_logs(EML_times, sol, mdl, DT_vars, user_constraints, objective_type,
               objective_var, model_name, log_path="/content/EML_results"):
    """
    TODO: Boscarini's function to be 1) refactored for printing also logic constraints, 2) cleaned and 3) documented.
    :param string:
    :param log_file:
    :return:
    """
    #### Log TIMES
    EML_times['after_solve_time'] = time.time()
    times = [EML_times['after_solve_time'] - EML_times['before_modelEM_time']]
    labels = ['tot_solve_time']
    for time_label in EML_times:
        labels.append(time_label)
        times.append(f"{EML_times[time_label] - EML_times['before_modelEM_time']}")
    times.append(mdl.solve_details.time)
    labels.append('CPLEX_time(sol)')
    with open(f"{log_path}/time_logs.csv", mode='a+') as results_file:
        results_writer = csv.writer(results_file, delimiter=',', quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
        results_writer.writerow(labels)
        results_writer.writerow(times)

    #### Log DATA & SOLUTION
    objective_str = f"{objective_type}({objective_var})"
    user_constraints_strs = []
    for i in range(len(user_constraints['variable'])):
        var_name = user_constraints['variable'][i]
        cstr_type = user_constraints['type'][i]
        v = user_constraints['value'][i]
        user_constraints_strs.append(f'{var_name}{cstr_type}{v}')
    sols = [objective_str, user_constraints_strs, model_name]
    labels = ['objective', 'constraints', 'model_file']

    # solution
    if sol is None:
        sols = sols + ['No sol found']
        labels = labels + ['status']
    else:
        sols = sols + [f"{sol.solve_details.status}", '{:.2f}'.format(
            mdl.solve_details.time)]
        labels = labels + ['status', 'time']
        for var in DT_vars:
            labels.append(var)
            sols.append(f"{sol[var]}")

    with open(f"{log_path}/sol_logs.csv", mode='a+') as results_file:
        results_writer = csv.writer(results_file, delimiter=',', quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
        results_writer.writerow(labels)
        results_writer.writerow(sols)


def print_log(string, log_file='log_shell.txt'):
    print(string)
    with open(log_file, mode='a+') as f:
        f.write(string + '\n')
