import copy
import pandas as pd
from utils.perfmetric_simple import perfmetric_simple
import concurrent.futures
from concurrent.futures import Future
from utils.utils_general import printProgressBar
import matplotlib.pyplot as plt
import numpy as np
import os


def ind_histogram(agent_dicts_dict, ind_range, PEND_EXEC, ENTRY_EXIT, MULTI_PROCESSING, print_progress=1):
    ### Perform evaluation & perf metric by ind range
    if MULTI_PROCESSING:
        pr_result_hist = []
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=int(os.cpu_count()*MULTI_PROCESSING))
        for il in ind_range:
            ih = il + ind_range[1] - ind_range[0]
            pr_result_hist.append(executor.submit(eval_perf_in_range, agent_dicts_dict, il, ih, PEND_EXEC, ENTRY_EXIT))

        ### Wait pr_result_hist all done and print
        ind_len = len(pr_result_hist)
        done_cnt = 0
        if print_progress==1:
            printProgressBar(done_cnt, ind_len, prefix = 'Indicator range:', suffix = 'Complete', length = 50)
        while True:
            someone_not_done = False
            for idx, future in enumerate(pr_result_hist):
                if isinstance(future, Future):
                    if future.done():
                        pr_result_hist[idx] = future.result()
                        done_cnt+=1
                        if print_progress==1:
                            printProgressBar(done_cnt, ind_len, prefix = 'Indicator range:', suffix = 'Complete', length = 50)
                    else:
                        someone_not_done = True
            if not someone_not_done:
                break
    else:
        pr_result_hist = []
        for idx, il in enumerate(ind_range):
            if print_progress==1:
                printProgressBar(idx, len(ind_range), prefix = 'Indicator range:', suffix = 'Complete', length = 50)
            ih = il + ind_range[1] - ind_range[0]
            pr_result_hist.append(eval_perf_in_range(agent_dicts_dict, il, ih, PEND_EXEC, ENTRY_EXIT))

    return pd.DataFrame(pr_result_hist)


def eval_perf_in_range(agent_dicts_dict, il, ih, pend_exec, entry_exit):
    agent_dicts_dict_copy = copy.deepcopy(agent_dicts_dict)
    ### Filter order in the range
    for agent_dict in agent_dicts_dict_copy.values():
        orders_info = agent_dict['orders_info']
        orders_info_noc = orders_info.loc[~(orders_info['cancel']==1)].reset_index(drop = True)
        orders_info_noc_pair =orders_info_noc.groupby('position_cnt').filter(lambda x: len(x) >= 2)

        ### Pick orders
        if entry_exit==1:
            orders_info_fil_pos = orders_info_noc_pair[(orders_info_noc_pair['base']==0)]
        else:
            orders_info_fil_pos = orders_info_noc_pair[(orders_info_noc_pair['base']!=0)]

        if pend_exec==1:
            orders_info_fil_type = orders_info_fil_pos[(orders_info_fil_pos['exec_ind']>=il) & (orders_info_fil_pos['exec_ind']<ih)]
        else:
            orders_info_fil_type = orders_info_fil_pos[(orders_info_fil_pos['pending_ind']>=il) & (orders_info_fil_pos['pending_ind']<ih)]

        pos_cnts_fil = orders_info_fil_type['position_cnt'].values.tolist()
        agent_dict['orders_info'] = orders_info[orders_info['position_cnt'].isin(pos_cnts_fil)].copy().reset_index(drop=True)

    ### Run performance metric calculation
    _, pr_result = perfmetric_simple(agent_dicts_dict_copy)
    pr_result.pop('eq_series', None)

    return pr_result