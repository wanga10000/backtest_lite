import os
from utils.utils_general import printProgressBar
from c_utils.c_backtest_core import c_backtest_core
from c_utils.c_backtest_mea_core import c_backtest_mea_core
import concurrent.futures
from concurrent.futures import Future


def backtest(agent_dicts_dict, MULTI_PROCESSING, print_progress=0):
    ### Run backtest loop
    agent_dicts_dict_return = dict()

    if len(agent_dicts_dict)==1:
        for agent_name, agent_dict in agent_dicts_dict.items():
            if agent_dict['backtest']['DETAIL_MEA']==1:
                agent_dicts_dict_return[agent_name] = c_backtest_mea_core(agent_dict, print_progress)
            else:
                agent_dicts_dict_return[agent_name] = c_backtest_core(agent_dict, print_progress)
        return agent_dicts_dict_return

    else:
        if MULTI_PROCESSING!=0:
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=int(os.cpu_count()*MULTI_PROCESSING))
            for agent_name, agent_dict in agent_dicts_dict.items():
                if agent_dict['backtest']['DETAIL_MEA']==1:
                    agent_dicts_dict_return[agent_name] = executor.submit(c_backtest_mea_core, agent_dict, 0)
                else:
                    agent_dicts_dict_return[agent_name] = executor.submit(c_backtest_core, agent_dict, 0)

            ### Wait agent_dict_list_return all done and print
            agent_len = len(agent_dicts_dict_return)
            done_cnt = 0
            if print_progress==1:
                printProgressBar(done_cnt, agent_len, prefix = 'Backtest:', suffix = 'Complete', length = 50)
            while True:
                someone_not_done = False
                for agent_name, future in agent_dicts_dict_return.items():
                    if isinstance(future, Future):
                        if future.done():
                            agent_dicts_dict_return[agent_name] = future.result()
                            done_cnt+=1
                            if print_progress==1:
                                printProgressBar(done_cnt, agent_len, prefix = 'Backtest:', suffix = 'Complete', length = 50)
                        else:
                            someone_not_done = True
                if not someone_not_done:
                    break
        else:
            agent_len = len(agent_dicts_dict)
            for aidx, (agent_name, agent_dict) in enumerate(agent_dicts_dict.items()):
                if print_progress==1:
                    printProgressBar(aidx, agent_len, prefix = 'Backtest:', suffix = 'Complete', length = 50)

                if agent_dict['backtest']['DETAIL_MEA']==1:
                    agent_dicts_dict_return[agent_name] = c_backtest_mea_core(agent_dict, 0)
                else:
                    agent_dicts_dict_return[agent_name] = c_backtest_core(agent_dict, 0)

    return agent_dicts_dict_return