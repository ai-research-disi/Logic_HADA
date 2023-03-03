# -*- coding: utf-8 -*-
"""
Created on Thu Mar 2 16:37:45 2023

@author: EleMisi
"""

import datetime
import sys
import time

import docplex.mp.model as cpx
import regex
from eml.backend import cplex_backend

from utils.util_functions import print_log, get_linear_expression, write_logs


def build_and_solve_EML(df_mins, df_maxs, user_constraints, logic_constraints,
                        objective_type, objective_var,
                        model_name=None, save_path='.',
                        enable_var_type=False, inst_descr=None, ml_trgt=None, algs=None):
    f = open('../vars_constr_num.txt', 'w')

    if not model_name:
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y-%H_%M_%S")
        model_name = f"EML_model_{dt_string}"
    print_log(f' ------------------- MODEL {model_name} -------------------')

    '''Section: Build & solve CPLEX Model'''
    print_log('\n=== Building basic model')
    EML_times = {}
    EML_times['before_modelEM_time'] = time.time()

    # Build a backend object
    bkd = cplex_backend.CplexBackend()
    # Build a docplex model
    mdl = cpx.Model()
    f.write('MARKER 0:model_creation:{}:{}\n'.format(mdl.number_of_constraints,
                                                     mdl.number_of_variables))
    ###### Define problem variables #####
    # DT variables
    DT_vars = []
    DT_vars_int = []

    # DT_vars_names = common_cols
    DT_vars_names_in = inst_descr
    DT_vars_names_out = ml_trgt
    DT_vars_names_params = []

    DT_vars_int_names = []
    binary_vars_names = []

    for alg in algs.keys():
        for var in algs[alg]['alg_params'].values():
            DT_vars_names_params.append(var['name'])
            if var['type'] == int:
                DT_vars_int_names.append(var['name'])

    bin_list = []
    # Algorithm (ANTICIPATE or CONTINGENCY) is stored as a binary var
    for alg in algs.keys():
        bin_list.append(mdl.binary_var(f"b_{alg}"))
        binary_vars_names.append(f"b_{alg}")

    # The sum of the binary algorithm variables must be equal to 1, so only 1 algorithm can be chosen (ANT or CONT)
    mdl.add_constraint(mdl.sum(bin_list) == 1)

    # Insert continuous variables for ML targets (e.g., 'memAvg(MB)', 'time(sec)', 'sol(keuro)') indexed via the algorithm
    # each variable is called 'y_{ALG_NAME}_{VAR_NAME}' and its upper and lower bound are stored
    # NB: both the upper and lower bound are assumed to be greater than zero
    print_log("DTs variables")
    print_log("\tIndexed via alg")
    for var in DT_vars_names_out:
        print_log(f"\t* {var}")
        for alg in algs.keys():
            print_log(f"\t\t* {alg}")
            DT_vars.append(mdl.continuous_var(lb=df_mins[alg].loc[var],
                                              ub=df_maxs[alg].loc[var], name=f"y_{alg}_{var}"))
            print_log(f"\t\cstr_type * lb = {df_mins[alg].loc[var]}")
            print_log(f"\t\cstr_type * ub = {df_maxs[alg].loc[var]}")

    # Insert continuous variables for ML input features, which form the instance description (e.g., 'PV_mean', 'PV_std', 'Load_mean', 'Load_std')
    # each variable is called 'y_{VAR_NAME}' and its upper and lower bound are stored
    # NB: both the upper and lower bound are assumed to be greater than zero
    print_log("\tInstance description")
    for var in DT_vars_names_in:
        print_log(f"\t* {var}")
        DT_vars.append(mdl.continuous_var(lb=df_mins['glob'].loc[var],
                                          ub=df_maxs['glob'].loc[var], name=f"y_{var}"))

    # Insert continuous variables for algorithm hyperparameters (e.g., 'nScenarios' or 'nTraces')
    # each variable is called 'y_{VAR_NAME}' and its upper and lower bound are stored
    # NB: both the upper and lower bound are assumed to be greater than zero
    #
    print_log("\tHparams")
    for var in DT_vars_names_params:
        print_log(f"\t* {var}")
        DT_vars.append(mdl.continuous_var(lb=df_mins['glob'].loc[var],
                                          ub=df_maxs['glob'].loc[var], name=f"y_{var}"))
        # TODO: for the moment, we do not force the continuous variable to be equal to the discrete one
        # by setting enable_var_type = False in the function argument
        if var in DT_vars_int_names and enable_var_type:
            DT_vars_int.append(mdl.integer_var(lb=df_mins['glob'].loc[var],
                                               ub=df_maxs['glob'].loc[var], name=f"y_{var}_int"))
            mdl.add_constraint(mdl.get_var_by_name(f'y_{var}_int') ==
                               mdl.get_var_by_name(f'y_{var}'))

    f.write('MARKER 2:after_DT_vars:{}:{}\n'.format(mdl.number_of_constraints,
                                                    mdl.number_of_variables))

    # Insert logic rules as indicator constraints
    print_log('\n=== Adding logic rules constraints')
    print_log("Logic rules constraints:")

    logicRules_vars = {}
    # Loop on the algorithms
    for alg in algs.keys():
        print_log(f"\t* {alg}")
        model = list(algs[alg]['ml_model'].keys())[0]
        print_log(f"\t\t with model {model}")
        logicRules_vars[alg] = []
        # Loop on the associated logic rules
        for i, lg_constr in enumerate(logic_constraints[alg][model]):
            if_constraint = lg_constr['if']
            then_constraint = lg_constr['then']
            # Add binary variables and constraints for the IF statement
            logicRules_vars[alg].append({'if': []})
            # Loop on the IF components
            for j in range(len(if_constraint['variable'])):
                logicRules_vars[alg][i]['if'].append(mdl.binary_var(name=f"{alg}_LogRul_{i}_IF_z{j}"))
                binary_vars_names.append(f"{alg}_LogRul_{i}_IF_z{j}")

                var_name = if_constraint['variable'][j]
                cstr_type = if_constraint['type'][j]
                value = if_constraint['value'][j]
                print_log(f'\t\t\t* IF {var_name} {cstr_type} {value}')

                # In GridREx IF statement there are only range constraints,
                # thus, we consider only this type of linear constraint
                if cstr_type == 'range':
                    var = mdl.get_var_by_name(name=var_name)
                    range_lb, range_ub = value

                    mdl.add_indicator(logicRules_vars[alg][i]['if'][j],
                                      var >= range_lb)
                    mdl.add_indicator(logicRules_vars[alg][i]['if'][j],
                                      var <= range_ub)

            # Add binary variables for the THEN statement
            logicRules_vars[alg][i]['then'] = mdl.binary_var(name=f"{alg}_LogRul_{i}_THEN_z{j + 1}")
            binary_vars_names.append(f"{alg}_LogRul_{i}_THEN_z{j + 1}")

            # Add indicator constraint to link IF and THEN statements
            # (If z_THEN == 1, then the binary variables of the IF statement are equal to 1)
            mdl.add_indicator(logicRules_vars[alg][i]['then'], mdl.sum(logicRules_vars[alg][i]['if']) == j + 1)

            # Loop on the THEN components
            for k in range(len(then_constraint['variable'])):

                var_name = then_constraint['variable'][k]
                cstr_type = then_constraint['type'][k]
                value = then_constraint['value'][k]
                print_log(f'\t\t\t\t THEN {var_name} {cstr_type} {value}')

                # In GridREx THEN statement there are only equality constraints,
                # thus, we consider only this type of linear constraint
                if cstr_type == '==':
                    var = mdl.get_var_by_name(name=var_name)
                    linear_expr = get_linear_expression(value)

                    mdl.add_indicator(logicRules_vars[alg][i]['then'],
                                      var == eval(linear_expr))

        # Only one of the logic rules must be true
        list_of_THEN_variables = [logicRules_vars[alg][i]['then'] for i in range(len(logicRules_vars[alg]))]
        mdl.add_constraint(mdl.sum(list_of_THEN_variables) == 1)

    ###### Define problem constraints & objective #####
    print_log('\n=== Adding custom constraints & objective')

    # Custom constraints
    print_log("Custom constraints:")
    for i in range(len(user_constraints['variable'])):
        var_name = user_constraints['variable'][i]
        cstr_type = user_constraints['type'][i]
        v = user_constraints['value'][i]
        print_log(f'\t* {var_name} {cstr_type} {v}')

        # we linearize the quadratic constraints (see commented first formulation)
        # using the suggestion found here:
        # http://yetanothermathprogrammingconsultant.blogspot.com/2008/05/multiplication-of-continuous-and-binary.html.
        # We assume that x_lo and x_up (y_{alg}_{var_name}.lb and y_{alg}_{var_name}.ub) are
        # greater than zero
        for alg in algs.keys():
            print_log(f"\t* {alg}")
            if cstr_type == '<=':
                # mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_{var_name}') *
                #    mdl.get_var_by_name(f'b_{alg}') <= v)
                mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_{var_name}') <= v)
                mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_{var_name}').lb *
                                   mdl.get_var_by_name(f'b_{alg}') <= v)

            elif cstr_type == '>=':
                # mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_{var_name}') *
                #    mdl.get_var_by_name(f'b_{alg}') >= v)
                mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_{var_name}') >= v)

            elif cstr_type == '==':
                mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_{var_name}').lb *
                                   mdl.get_var_by_name(f'b_{alg}') <= v)
                mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_{var_name}').ub *
                                   mdl.get_var_by_name(f'b_{alg}') >= v)
                mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_{var_name}') -
                                   mdl.get_var_by_name(f'y_{alg}_{var_name}').ub *
                                   (1 - mdl.get_var_by_name(f'b_{alg}')) <= v)
                mdl.add_constraint(mdl.get_var_by_name(f'y_{alg}_{var_name}') -
                                   mdl.get_var_by_name(f'y_{alg}_{var_name}').lb *
                                   (1 - mdl.get_var_by_name(f'b_{alg}')) >= v)
            else:
                print('Unsupported constraint type, terminating..')
                sys.exit()

    # Objective
    print_log("Objective:")

    # build a list of expressions y_{alg}_cost * b_{alg}
    prod_list = []
    for alg in algs.keys():
        prod_list.append(mdl.get_var_by_name(f'y_{alg}_sol(keuro)') *
                         mdl.get_var_by_name(f'b_{alg}'))

    print_log(f'\t* {objective_type}')
    if objective_type == 'min':
        mdl.minimize(mdl.sum(prod_list))
    else:
        mdl.maximize(mdl.sum(prod_list))

    EML_times['after_settings_time'] = time.time()

    '''
    with open('obtained_model.txt', "w") as output_file:
        output_file.write(mdl.export_as_lp_string())
    '''
    ##### Print info & save EML model #####
    print_log('\n=== Print info & save')
    vars = mdl.find_re_matching_vars(regex.compile(r'.*'))
    print_log(f'{len(vars)} VARIABLES')
    for var in mdl.find_re_matching_vars(regex.compile(r'^((?!DT).)*$')):
        print_log(f"\t* {var}")

    ntot = 0
    for _ in mdl.generate_user_linear_constraints():
        ntot = ntot + 1
    print_log(f'{ntot} LINEAR CONSTRAINTS')
    for i, obj in enumerate(mdl.generate_user_linear_constraints()):
        string = f"\t* {obj}"
        if "DT" not in string:
            print_log(string)

    ntot = 0
    for _ in mdl.iter_indicator_constraints():
        ntot = ntot + 1
    print_log(f'{ntot} INDICATOR CONSTRAINTS')
    for i, obj in enumerate(mdl.iter_indicator_constraints()):
        string = f"\t* {obj}"
        if "DT" not in string:
            print_log(string)

    EML_times['after_print_time'] = time.time()

    # save model to be reused
    model_path = f'{save_path}/{model_name}'
    mdl.export_as_sav(model_path)
    print_log(f"\nModel saved to: {model_path}")

    EML_times['after_save_time'] = time.time()

    ## Print time data
    EML_times['after_modelEM_time'] = time.time()
    tot_time = EML_times['after_modelEM_time'] - EML_times['before_modelEM_time']
    print_log(f"\nTotal time needed to create MP model {tot_time}")
    for time_label in EML_times:
        t_time = EML_times[time_label] - EML_times['before_modelEM_time']
        print_log(f"* {time_label}: {t_time}")

    ################################ Solve ################################
    print_log('\n=== Starting the solution process')
    mdl.set_time_limit(20000)
    sol = mdl.solve()

    # Print solution
    if sol is None:
        print_log('No solution found')
    else:
        print_log('SOLUTION DATA')
        print_log('Solution time: {:.2f} (sec)'.format(mdl.solve_details.time))
        print_log('Solver status: {}'.format(sol.solve_details.status))

        print_log(f'\t*CONT VARIABLES')
        for var in DT_vars:
            print_log(f'\t\t* {var}: {sol[var]}')

        print_log(f'\t*INT VARIABLES')
        for var in DT_vars_int:
            print_log(f'\t\t* {var}: {sol[var]}')

        print_log(f'\t*BINARY VARIABLES')
        for var in binary_vars_names:
            print_log(f'\t\t* {var}: {sol[var]}')

    # Log
    DT_vars = DT_vars + DT_vars_int
    write_logs(EML_times=EML_times,
               sol=sol,
               mdl=mdl,
               DT_vars=DT_vars,
               user_constraints=user_constraints,
               objective_type=objective_type,
               objective_var=objective_var,
               model_name=model_name,
               log_path=save_path)

    print_log(f'----------------------------------------------------------')
