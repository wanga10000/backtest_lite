import itertools
from utils.backtest import backtest
from utils.perfmetric_simple import perfmetric_simple
import pandas as pd
import numpy as np
import concurrent.futures
from concurrent.futures import Future
from copy import deepcopy
import os
from datetime import timedelta
from math import log
from itertools import combinations
from utils.utils_general import printProgressBar


def pbo(agent_dicts_dict, SEP, METRICS, MULTI_PROCESSING, print_progress=0):
    agent_dicts_dict_copy = deepcopy(agent_dicts_dict)

    ### Initialize executor
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=int(os.cpu_count()*MULTI_PROCESSING))

    ### Loop agent and add job
    # Split paras name and range
    opt_params = next(iter(agent_dicts_dict_copy.values()))['opt_params']
    opt_params = {para: para_args for para, para_args in opt_params.items() if para not in ['AGENT', 'AUGMENT', 'AUGMENT_INC_ORG']}
    opt_params_range = []
    opt_params_names = []
    for key, value in opt_params.items():
        [en, mint, maxt, step] = list(map(float, str(value).split(",")))
        if en!=0:
            opt_params_range.append([mint, maxt, step])
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
    orders_infos_dicts_list = []   # The first dict is by agent, the second list is opt params
    for opt_params in opt_params_list:
        orders_infos_dicts_list.append(executor.submit(pbo_backtest_core, agent_dicts_dict, opt_params))

    ### Wait final_result all done and print
    done_target = len(orders_infos_dicts_list)
    done_cnt = 0
    if print_progress==1:
        printProgressBar(done_cnt, done_target, prefix = 'PBO stage1:', suffix = 'Complete', length = 50)
    while True:
        someone_not_done = False
        for idx, future_tmp in enumerate(orders_infos_dicts_list):
            if isinstance(future_tmp, Future):
                if future_tmp.done():
                    orders_infos_dicts_list[idx] = future_tmp.result()
                    done_cnt+=1
                    if print_progress==1:
                        printProgressBar(done_cnt, done_target, prefix = 'PBO stage1:', suffix = 'Complete', length = 50)
                else:
                    someone_not_done = True
        if not someone_not_done:
            break

    # Find earlest start and latest stop
    agent_dict_info = next(iter(agent_dicts_dict.values()))
    start_date = agent_dict_info['backtest']['START_DATE'].to_pydatetime()
    stop_date = agent_dict_info['backtest']['STOP_DATE'].to_pydatetime()
    for agent_dict in agent_dicts_dict.values():
        start_date_tmp = agent_dict['backtest']['START_DATE'].to_pydatetime()
        stop_date_tmp = agent_dict['backtest']['STOP_DATE'].to_pydatetime()
        start_date = min(start_date, start_date_tmp)
        stop_date = max(stop_date, stop_date_tmp)

    # Find divided dates
    divided_dates = [start_date + timedelta(days=((stop_date - start_date).days//SEP) * i) for i in range(SEP)]
    divided_dates.append(stop_date)

    # Make divided dataset
    orders_infos_dicts_list_s_date = []
    for didx, date in enumerate(divided_dates):
        if didx == len(divided_dates)-1:
            break
        else:
            nxt_date = divided_dates[didx+1]
            orders_infos_dicts_list_tmp = []
            for orders_infos_dict in orders_infos_dicts_list:
                orders_infos_dict_tmp = dict()
                for agent_name, orders_info in orders_infos_dict.items():
                    orders_info_sep = orders_info.loc[(orders_info['pending_date'] >= date) & (orders_info['pending_date'] < nxt_date)]
                    orders_infos_dict_tmp[agent_name] = orders_info_sep
                orders_infos_dicts_list_tmp.append(orders_infos_dict_tmp)
            orders_infos_dicts_list_s_date.append(orders_infos_dicts_list_tmp)

    # Start pbo loop
    pbo_sel_list = list(combinations(range(SEP), SEP//2))
    pbo_sel_list = [list(pbo_sel) for pbo_sel in pbo_sel_list]

    lambda_dict_list = []
    for pbo_sel in pbo_sel_list:
        lambda_dict_list.append(executor.submit(pbo_sel_calc_core, agent_dicts_dict, pbo_sel, orders_infos_dicts_list_s_date, METRICS))

    ### Wait final_result all done and print
    done_target = len(lambda_dict_list)
    done_cnt = 0
    if print_progress==1:
        printProgressBar(done_cnt, done_target, prefix = 'PBO stage2:', suffix = 'Complete', length = 50)
    while True:
        someone_not_done = False
        for idx, future_tmp in enumerate(lambda_dict_list):
            if isinstance(future_tmp, Future):
                if future_tmp.done():
                    lambda_dict_list[idx] = future_tmp.result()
                    done_cnt+=1
                    if print_progress==1:
                        printProgressBar(done_cnt, done_target, prefix = 'PBO stage2:', suffix = 'Complete', length = 50)
                else:
                    someone_not_done = True
        if not someone_not_done:
            break

    # lambda list of dict to dict of list
    lambda_list_dict = pd.DataFrame(lambda_dict_list).to_dict('list')

    # Calculate PBO for each metric(Count how many negative value in lambda list)
    pbo_dict = dict()
    for metric, lambda_list in lambda_list_dict.items():
        pbo_dict[metric] = (sum([1 for i in lambda_list if i < 0])) / len(lambda_list)

    return pbo_dict


def pbo_backtest_core(agent_dicts_dict, opt_params):
    ### Replace agent stra with opt_params
    for key, value in opt_params.items():
        for agent_dict in agent_dicts_dict.values():
            if key not in agent_dict['strategy'].keys():
                raise ValueError(key + " is not available")
            agent_dict['strategy'][key] = value

    ### Backtesting
    agent_dicts_dict = backtest(agent_dicts_dict, MULTI_PROCESSING=0)

    return {agent_name: agent_dict['orders_info'] for agent_name, agent_dict in agent_dicts_dict.items()}


def pbo_sel_calc_core(agent_dicts_dict, pbo_sel, orders_infos_dicts_list_s_date, METRICS):
    # Separate training, testing set
    orders_infos_dicts_list_train_s_date = [orders_infos_dicts_list_s_date[i] for i in pbo_sel]
    orders_infos_dicts_list_test_s_date = [orders_infos_dicts_list_s_date[i] for i in range(len(orders_infos_dicts_list_s_date)) if i not in pbo_sel]

    # Combine orders for training, testing set
    orders_infos_dicts_list_train = []
    orders_infos_dicts_list_test = []

    for vidx in range(len(orders_infos_dicts_list_s_date[0])):
        # Train process
        orders_infos_dict_tmp = deepcopy(orders_infos_dicts_list_train_s_date[0][vidx])
        for tidx in range(len(orders_infos_dicts_list_train_s_date)):
            if tidx==0:
                continue
            for agent_name in orders_infos_dict_tmp.keys():
                orders_infos_dict_tmp[agent_name] = pd.concat([orders_infos_dict_tmp[agent_name], orders_infos_dicts_list_train_s_date[tidx][vidx][agent_name]])
        orders_infos_dicts_list_train.append(orders_infos_dict_tmp)

        # Test process
        orders_infos_dict_tmp = deepcopy(orders_infos_dicts_list_test_s_date[0][vidx])
        for tidx in range(len(orders_infos_dicts_list_test_s_date)):
            if tidx==0:
                continue
            for agent_name in orders_infos_dict_tmp.keys():
                orders_infos_dict_tmp[agent_name] = pd.concat([orders_infos_dict_tmp[agent_name], orders_infos_dicts_list_test_s_date[tidx][vidx][agent_name]])
        orders_infos_dicts_list_test.append(orders_infos_dict_tmp)

    # Calculate all training, testing sets possible metric result
    # Train
    train_results_dict = dict()
    for metric in METRICS:
        train_results_dict[metric] = []

    for train_orders_dict in orders_infos_dicts_list_train:
        agent_dicts_dict_fake = deepcopy(agent_dicts_dict)
        for agent_name in agent_dicts_dict_fake.keys():
            agent_dicts_dict_fake[agent_name]['orders_info'] = train_orders_dict[agent_name]
        _, pr_result = perfmetric_simple(agent_dicts_dict_fake)

        for metric in METRICS:
            train_results_dict[metric].append(pr_result[metric])

    # Test
    test_results_dict = dict()
    for metric in METRICS:
        test_results_dict[metric] = []

    for test_orders_dict in orders_infos_dicts_list_test:
        agent_dicts_dict_fake = deepcopy(agent_dicts_dict)
        for agent_name in agent_dicts_dict_fake.keys():
            agent_dicts_dict_fake[agent_name]['orders_info'] = test_orders_dict[agent_name]
        _, pr_result = perfmetric_simple(agent_dicts_dict_fake)

        for metric in METRICS:
            test_results_dict[metric].append(pr_result[metric])

    # Find the best paras in training would get what rank in test results, for each metric
    lambda_calc_dict = dict()
    for metric in METRICS:

        best_idx = train_results_dict[metric].index(max(train_results_dict[metric]))
        rank_w = (sorted(test_results_dict[metric]).index(test_results_dict[metric][best_idx])+1)/len(test_results_dict[metric])
        lambda_calc_dict[metric] = 100 if rank_w==1 else log(rank_w/(1-rank_w))

    return lambda_calc_dict
