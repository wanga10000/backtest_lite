import itertools
from utils.utils_general import augment_str_decode, read_df_sets_add_atr_noise
from utils.backtest import backtest
from utils.perfmetric import perfmetric
import pandas as pd
import numpy as np
import concurrent.futures
from concurrent.futures import Future
from copy import deepcopy
import random
import os
import math


def test_multiple(agent_dicts_dict, MULTI_PROCESSING, MEAN_PR=0, print_done=0):
    agent_dicts_dict_copy = deepcopy(agent_dicts_dict)

    ### Initialize executor
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=int(os.cpu_count()*MULTI_PROCESSING))

    ### Handle atr noise df
    augment, _ = augment_str_decode(next(iter(agent_dicts_dict_copy.values()))['opt_params']['AUGMENT'])
    atr_gens_dict = None
    if any(d.get('method') == 'ATR_NOISE' for d in augment):
        atr_gens_dict = dict()
        aug_dict = next(d for d in augment if d.get('method') == 'ATR_NOISE')
        for agent_name, agent_dict in agent_dicts_dict_copy.items():
            df_dicts_list = read_df_sets_add_atr_noise(agent_dict, gen_num=aug_dict['repeats'], atr_noise=aug_dict['atr_noise'])
            atr_gens_dict[agent_name] = df_dicts_list

    ### Loop agent and add job
    # Split paras name and range
    opt_params = next(iter(agent_dicts_dict_copy.values()))['opt_params']
    opt_params = {para: para_args for para, para_args in opt_params.items() if para not in ['AGENT', 'AUGMENT', 'AUGMENT_INC_ORG']}
    opt_params_range = []
    opt_params_names = []
    for key, value in opt_params.items():
        [en, min, max, step] = list(map(float, str(value).split(",")))
        if en!=0:
            opt_params_range.append([min, max, step])
            opt_params_names.append(key)

    # Creating the initial paras
    range_list = []
    for i in range(len(opt_params_range)):
        opt_params_range[i][2] += 0.00000001
        range_list.append(np.arange(*opt_params_range[i]))

    paras_combs = np.array(list(itertools.product(*range_list)))
    paras_combs = np.matrix.round(paras_combs, 5)
    opt_params_list = [dict(zip(opt_params_names, paras)) for paras in paras_combs]

    # Loop Var paras by test multiple core
    pr_results_lists_list = []   # The first list is augment, the second list if opt params
    for opt_params in opt_params_list:
        pr_results_lists_list.append(test_multiple_core(agent_dicts_dict_copy, opt_params, MEAN_PR, executor, atr_gens_dict))

    ### Wait final_result all done and print
    while True:
        someone_not_done = False
        for idx1, pr_results_lists in enumerate(pr_results_lists_list):
            for idx0, pr_result_future in enumerate(pr_results_lists):
                if isinstance(pr_result_future, Future):
                    if pr_result_future.done():
                        pr_results_lists[idx0] = pr_result_future.result()
                        if print_done==1:
                            print("Repeats:", idx0, "in opt_params:", opt_params_list[idx1], "done")
                    else:
                        someone_not_done = True
        if not someone_not_done:
            break

    ### Average 2nd level pr_result, make 1st result dataframe
    for idx, pr_results_lists in enumerate(pr_results_lists_list):
        pr_result_df = pd.DataFrame(pr_results_lists)
        pr_result_df = pr_result_df.drop('eq_series', axis=1) if 'eq_series' in pr_result_df.columns else pr_result_df
        pr_result = pr_result_df.mean().to_dict()
        pr_results_lists_list[idx] = pr_result

    return pd.DataFrame(opt_params_list), pd.DataFrame(pr_results_lists_list)


def test_multiple_core(agent_dicts_dict, opt_params, MEAN_PR, executor, atr_gens_dict):
    ### Replace agent stra with opt_params
    for key, value in opt_params.items():
        for agent_dict in agent_dicts_dict.values():
            if key not in agent_dict['strategy'].keys():
                raise ValueError(key + " is not available")
            agent_dict['strategy'][key] = value

    ### Augment
    augment, remove_top_per = augment_str_decode(next(iter(agent_dicts_dict.values()))['opt_params']['AUGMENT'])
    augment_inc_org = next(iter(agent_dicts_dict.values()))['opt_params']['AUGMENT_INC_ORG']
    agent_dicts_dicts_list = [deepcopy(agent_dicts_dict)] if augment_inc_org==1 else []
    for aug_dict in augment:
        if aug_dict['method']=='SHIFT_TIME':
            # Shift time method: Use TF_OFFSET to augment the dataset. Use shift_tf to shift
            repeats = aug_dict['repeats']
            shift_tf = aug_dict['shift_tf']
            tf_to_min_table = {'m': 1, 'h': 60, 'd': 1440}
            tf_to_hour_table = {'h': 1, 'd': 24}
            agent_dicts_dicts_list_tmp = []
            for agent_name, agent_dict in agent_dicts_dict.items():
                agent_dicts_dict_tmp = dict()

                # Find full offset list
                shift_tf_min = int(shift_tf[:-1])
                main_tf = agent_dict['strategy']['MAIN_TF']
                main_tf_min = int(main_tf[:-1]) * tf_to_hour_table[main_tf[-1]] if shift_tf[-1]=='h' else int(main_tf[:-1]) * tf_to_min_table[main_tf[-1]]
                full_offset_list = list(range(0, main_tf_min, shift_tf_min))

                # Sample offset
                if repeats >= len(full_offset_list):
                    sample_offset_list = list(itertools.islice(itertools.cycle(full_offset_list), repeats))
                else:
                    sample_offset_list = random.sample(full_offset_list, repeats)
                sample_offset_list = [str(offset) + shift_tf[-1] for offset in sample_offset_list]

                # Sample offset
                for offset in sample_offset_list:
                    agent_dict_tmp = deepcopy(agent_dict)
                    used_tfs = str(agent_dict_tmp['general']['USED_TFS']).split(",")
                    tf_offset = str(agent_dict_tmp['general']['TF_OFFSET']).split(",")
                    main_tf_pos = used_tfs.index(str(agent_dict_tmp['strategy']['MAIN_TF']))
                    tf_offset[main_tf_pos] = offset
                    agent_dict_tmp['general']['TF_OFFSET'] = ','.join(tf_offset)
                    agent_dicts_dict_tmp[agent_name+'_'+offset] = agent_dict_tmp

                # Append to agent_dicts_dicts_list_tmp
                if len(agent_dicts_dicts_list_tmp)==0:
                    agent_dicts_dicts_list_tmp = [{agent_name_tmp: agent_dict_tmp} for agent_name_tmp, agent_dict_tmp in agent_dicts_dict_tmp.items()]
                else:
                    for i, (agent_name_tmp, agent_dict_tmp) in enumerate(agent_dicts_dict_tmp.items()):
                        agent_dicts_dicts_list_tmp[i][agent_name_tmp] = agent_dict_tmp

        elif aug_dict['method']=='ATR_NOISE':
            # ATR noise method: Use ATR to add noise
            agent_dicts_dicts_list_tmp = [dict() for _ in range(int(aug_dict['repeats']))]
            for agent_name, agent_dict in agent_dicts_dict.items():
                # Make several df_sets added with atr_noise
                for idx in range(aug_dict['repeats']):
                    agent_dict_tmp = deepcopy(agent_dict)
                    df_dicts = atr_gens_dict[agent_name][idx]
                    agent_dict_tmp['df_sets'] = df_dicts['df_sets']
                    agent_dict_tmp['tick_df_sets'] = df_dicts['tick_df_sets']
                    agent_dicts_dicts_list_tmp[idx][agent_name+'_'+"a"+str(idx)] = agent_dict_tmp
        else:
            raise ValueError("No such augment method")

        # Append to agent_dicts_dict_list
        agent_dicts_dicts_list = agent_dicts_dicts_list_tmp

    ### Backtesting + Evaluation + Perf_metric
    pr_results_list = []
    for agent_dicts_dict in agent_dicts_dicts_list:
        pr_results_list.append(executor.submit(back_perf, agent_dicts_dict, remove_top_per, MEAN_PR))

    return pr_results_list


def back_perf(agent_dicts_dict, remove_top_per, MEAN_PR):
    agent_dicts_dict = backtest(agent_dicts_dict, MULTI_PROCESSING=0)

    # Remove top group
    if remove_top_per>0:
        for agent_name, agent_dict in agent_dicts_dict.items():
            # Find remove order
            orders_info = agent_dict['orders_info']
            orders_info_noc = orders_info.loc[~(orders_info['cancel']==1)].reset_index(drop = True)
            orders_info_noc_pair =orders_info_noc.groupby('position_cnt').filter(lambda x: len(x) >= 2)
            orders_info_noc_close_pos = orders_info_noc_pair.loc[orders_info_noc_pair['base']==0].copy()
            orders_info_noc_close_pos_over0 = orders_info_noc_close_pos[(orders_info_noc_close_pos['pnl']>0)].reset_index(drop = True).sort_values(by='pnl', ascending=False).reset_index(drop = True)
            remove_cnt = math.ceil(len(orders_info_noc_close_pos_over0)*remove_top_per*0.01)
            remove_pos_cnt_list = orders_info_noc_close_pos_over0['position_cnt'][:remove_cnt].to_list()

            # Remove order in orders_info
            agent_dict['orders_info'] = orders_info.loc[~(orders_info['position_cnt'].isin(remove_pos_cnt_list))].reset_index(drop = True)

            # Remove balance_eq
            if 'eq_series' in agent_dict.keys():
                agent_dict.pop('eq_series')
            agent_dicts_dict[agent_name] = agent_dict

    # pr_results by all or by mean
    if MEAN_PR==1:
        # Run pr for each agent
        pr_results_dict = dict()
        for agent_name, agent_dict in agent_dicts_dict.items():
            _, pr_results = perfmetric({agent_name: agent_dict})
            pr_results_dict[agent_name] = pr_results

        # Average pr_result and print, only include non-outlier
        metric_keys = [key for key in next(iter(pr_results_dict.values())).keys() if key!='eq_series']
        pr_results_out = dict()

        # For each key, calculate the average value
        for key in metric_keys:
            values = [d[key] for d in list(pr_results_dict.values())]
            pr_results_out[key] = sum(values) / len(values)
    else:
        _, pr_results_out = perfmetric(agent_dicts_dict)

    return pr_results_out
