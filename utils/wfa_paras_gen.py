import os, glob
from pathlib import Path
from utils.utils_general import delete_file
from copy import deepcopy
from utils.test_multiple import test_multiple
from datetime import datetime
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


def wfa_paras_gen(agent_dicts_dict, NEW_WFA, WFA_PATH, MULTI_PROCESSING, print_progress=0):
    ########## Make wfa folder
    if not os.path.exists(f"{WFA_PATH}"):
        if NEW_WFA==0:
            raise ValueError("NEW_WFA=0 without previous wfa paras")
        os.makedirs(f"{WFA_PATH}")
    else:
        if NEW_WFA==1:
            delete_file(WFA_PATH)
            os.makedirs(f"{WFA_PATH}")


    ########## Gen walk_forward weights
    ### loop for agent
    next_gen_date_dict = dict()
    for agent_name, agent_dict in agent_dicts_dict.items():
        if 'walk_forward' in agent_dict.keys():
            if NEW_WFA==0:
                ### Check previous wfa_info is met, or raise error
                pre_wfa_info = pd.read_excel(WFA_PATH + agent_dict['walk_forward']['AGENT'] + '/wfa_info.xlsx', sheet_name="wfa_info").to_dict('records')[0]
                pre_wfa_info['AGENT'] = agent_dict['walk_forward']['AGENT']
                pre_wfa_info['WFA_START_DATE'] = agent_dict['walk_forward']['WFA_START_DATE']
                pre_wfa_info['WFA_STOP_DATE'] = agent_dict['walk_forward']['WFA_STOP_DATE']
                if agent_dict['walk_forward']!=pre_wfa_info:
                    raise ValueError(agent_dict['walk_forward']['AGENT'] + " wfa info not met, cannot apply NEW_WFA=0")

                ### Get all previous wfa datetimes
                pre_wfa_dates = [Path(x).stem for x in glob.glob(WFA_PATH + agent_dict['walk_forward']['AGENT'] + '/[0-9]*')]

            else:
                ### Record wfa info
                wfa_info = deepcopy(agent_dict['walk_forward'])
                wfa_info.pop('AGENT')
                wfa_info.pop('WFA_START_DATE')
                wfa_info.pop('WFA_STOP_DATE')
                wfa_info = pd.Series(wfa_info).to_frame().T
                delete_file(WFA_PATH + agent_name + '/')
                os.makedirs(WFA_PATH + agent_name + '/')
                with pd.ExcelWriter(WFA_PATH + agent_name + '/wfa_info.xlsx') as writer:
                    wfa_info.to_excel(writer, index=False, header=True, sheet_name='wfa_info')

                pre_wfa_dates = []

            ### Initialize
            baseline_date = agent_dict['walk_forward']['BASELINE_DATE']
            step = agent_dict['walk_forward']['STEP']
            train_period = agent_dict['walk_forward']['TRAIN_PERIOD']
            metrics = agent_dict['walk_forward']['METRICS'].split(',')
            opt_method = agent_dict['walk_forward']['OPT_METHOD']
            if opt_method=='ALL' and len(metrics)!=1:
                raise ValueError("ALL Method should have only 1 metric")

            ### Find closest wfa datetime to current date
            # Find latest date in dataset
            latest_wfa_date = baseline_date
            while latest_wfa_date <= agent_dict['walk_forward']['WFA_STOP_DATE'] - relativedelta(days=step):
                latest_wfa_date += relativedelta(days=step)
            next_gen_date_dict[agent_name] = latest_wfa_date + relativedelta(days=step)

            ### Define wfa_dates
            start_date = agent_dict['walk_forward']['WFA_START_DATE'].to_pydatetime()
            stop_date = latest_wfa_date.to_pydatetime()
            wfa_dates = [stop_date]
            dt = stop_date
            # Search backward
            while (dt - relativedelta(days=step))>=(start_date + relativedelta(days=train_period)):
                dt -= relativedelta(days=step)
                wfa_dates.append(dt)

            ### for loop for walk forward and gen paras
            for dt in wfa_dates:
                if NEW_WFA==1 or dt.strftime("%Y-%m-%d-%H-%M-%S") not in pre_wfa_dates:
                    if print_progress==1:
                        print(agent_name + ": " + dt.strftime("%Y-%m-%d %H:%M:%S") + " Magic gen......")
                    train_start_date = dt - relativedelta(days=train_period)
                    agent_dict_copy = deepcopy(agent_dict)
                    agent_dict_copy['backtest']['START_DATE'] = pd.Timestamp(train_start_date)
                    agent_dict_copy['backtest']['STOP_DATE'] = pd.Timestamp(dt)
                    agent_dicts_dict_copy = {agent_name: agent_dict_copy}

                    # Find the best opt_params
                    if opt_method=='ALL':
                        # Test multiple on this group and datetime
                        opt_params_df, pr_results_df = test_multiple(agent_dicts_dict_copy, MULTI_PROCESSING, print_done=0)
                        pr_results_df_metric = pr_results_df[metrics[0]]
                        best_idx = pr_results_df_metric.idxmax()
                        if np.isnan(best_idx):
                            continue
                        best_var = opt_params_df.loc[best_idx].to_dict()

                    elif opt_method=='BY1VAR':
                        # Get en=1 opt_params list
                        opt_params_list = []
                        for var_name, var_option in agent_dict['opt_params'].items():
                            if var_name not in ['AGENT', 'AUGMENT', 'AUGMENT_INC_ORG']:
                                [en, _, _, _] = list(map(float, str(var_option).split(",")))
                                if en==1:
                                    opt_params_list.append(var_name)

                        # For loop for these opt_params to do BY1VAR
                        best_var = dict()
                        has_nan = 0
                        for vidx, var_name in enumerate(opt_params_list):
                            opt_params_new = deepcopy(agent_dict['opt_params'])

                            # Set other vars en to 0
                            for var_new_name, var_new_option in opt_params_new.items():
                                if var_new_name!=var_name and var_new_name not in ['AGENT', 'AUGMENT', 'AUGMENT_INC_ORG']:
                                    opt_params_new[var_new_name] = '0' + var_new_option[1:]

                            # Replace opt_params to agent_dicts_dict_copy
                            agent_dicts_dict_copy[agent_name]['opt_params'] = opt_params_new

                            # Do single var test_multiple
                            opt_params_df, pr_results_df = test_multiple(agent_dicts_dict_copy, MULTI_PROCESSING, print_done=0)
                            pr_results_df_metric = pr_results_df[metrics[vidx]]
                            best_idx = pr_results_df_metric.idxmax()
                            if np.isnan(best_idx):
                                has_nan = 1
                                break
                            best_var_by1 = opt_params_df.loc[best_idx].to_dict()

                            # Replace strategy in agent_dicts_dict_copy
                            agent_dicts_dict_copy[agent_name]['strategy'].update(best_var_by1)
                            best_var[var_name] = best_var_by1[var_name]

                        if has_nan==1:
                            continue
                    else:
                        raise ValueError("Unknown opt method")

                    # Replace the stra paras and store
                    new_strategy_cfg = deepcopy(agent_dict['strategy'])
                    new_strategy_cfg.update(best_var)
                    new_strategy_cfg = pd.Series(new_strategy_cfg).to_frame().T

                    # Store new_strategy_cfg
                    with pd.ExcelWriter(WFA_PATH + agent_name + '/' + datetime.strftime(dt, '%Y-%m-%d-%H-%M-%S') + '.xlsx') as writer:
                        new_strategy_cfg.to_excel(writer, index=False, header=True, sheet_name='strategy')
        else:
            raise ValueError("No wfa info define in config")

    return next_gen_date_dict
