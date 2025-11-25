from utils.backtest import backtest
from utils.perfmetric import perfmetric
from utils.utils_general import printProgressBar
import pandas as pd
import numpy as np
import math


def monte_carlo_trial(agent_dicts_dict, EPOCHS, REMOVE_TOP, MULTI_PROCESSING, SAMPLE_DAY = None, print_progress=0):
    ### Run backtest
    agent_dicts_dict = backtest(agent_dicts_dict, MULTI_PROCESSING=MULTI_PROCESSING, print_progress=print_progress)

    ### Make original performance metric 1by1
    pr_results_df_org_dict = dict()
    for agent_name, agent_dict in agent_dicts_dict.items():
        net_worth_df = get_net_worth_df(agent_dict['orders_info'])
        pr_results_df_org_dict[agent_name] = perfmetric_simple(net_worth_df)

    ### Loop for agent_dict
    pr_results_dfs_dict = dict()
    pidx = 0
    for agent_name, agent_dict in agent_dicts_dict.items():
        ### Count "sample_day" should be how many trades
        orders_info = agent_dict['orders_info']
        start_date = agent_dict['backtest']['START_DATE']
        stop_date = agent_dict['backtest']['STOP_DATE']
        if SAMPLE_DAY!=None:
            trade_num = int(len(set(orders_info['position_cnt'])) * (SAMPLE_DAY / (stop_date-start_date).days)) if SAMPLE_DAY<=(stop_date-start_date).days else None
        else:
            trade_num = None

        ### Monte carlo Loop epochs
        pr_results_list = []

        for _ in range(EPOCHS):
            if print_progress==1:
                printProgressBar(pidx, EPOCHS*len(agent_dicts_dict), prefix = 'Monte carlo:', suffix = 'Complete', length = 50)
                pidx+=1
            pr_results_list.append(monte_carlo_trial_core(agent_dict, REMOVE_TOP, trade_num))

        pr_results_dfs_dict[agent_name] = pd.DataFrame(pr_results_list)

    return pr_results_df_org_dict, pr_results_dfs_dict


def monte_carlo_trial_core(agent_dict, REMOVE_TOP, trade_num):
    # Sample trades
    net_worth_df = get_net_worth_df(agent_dict['orders_info'])
    if trade_num==None:
        trade_num = len(net_worth_df)

    net_worth_df = net_worth_df.sample(n=trade_num, replace=False).reset_index(drop=True)

    # Do perfmetric and return
    return perfmetric_simple(net_worth_df, REMOVE_TOP)


def get_net_worth_df(orders_info):
    ### Remove cancel and filter pair
    orders_info = orders_info.copy()
    orders_info_noc = orders_info.loc[~(orders_info['cancel']==1)].reset_index(drop = True)
    orders_info_noc_pair =orders_info_noc.groupby('position_cnt').filter(lambda x: len(x) >= 2)

    ### Collect normalized return df by group
    orders_info_noc_close_pos = orders_info_noc_pair.loc[orders_info_noc_pair['base']==0].copy()
    orders_info_noc_close_pos['margin'] = abs(orders_info_noc_pair.iloc[1::2]['exec_base'].values) * orders_info_noc_pair.iloc[::2]['exec_price'].values
    orders_info_noc_close_pos['net_worth'] = orders_info_noc_close_pos['pnl'] / orders_info_noc_close_pos['margin']
    net_worth_df = pd.DataFrame(dict(position_cnt = orders_info_noc_close_pos['position_cnt'], net_worth = orders_info_noc_close_pos['net_worth']))

    return net_worth_df


def perfmetric_simple(net_worth_df, REMOVE_TOP=0):
    ### Remove best n positions
    net_worth_df_sort = net_worth_df.reset_index(drop = True).sort_values(by='net_worth', ascending=False).reset_index(drop = True)
    remove_cnt = math.floor(len(net_worth_df)*REMOVE_TOP*0.01)
    remove_position_cnts = net_worth_df_sort['position_cnt'][:remove_cnt].to_list()
    net_worth_df = net_worth_df.loc[~(net_worth_df['position_cnt'].isin(remove_position_cnts))].reset_index(drop = True)

    ### Net worth
    net_worth = 1
    net_worth_list = [1]
    for nw in net_worth_df['net_worth']:
        net_worth *= (1 + nw)
        net_worth_list.append(net_worth)

    ### Max dd
    eq_array = np.array(net_worth_list)
    cumulative_max = np.maximum.accumulate(eq_array)
    drawdowns = (cumulative_max - eq_array) / cumulative_max
    max_drawdown = np.max(drawdowns)
    mean_drawdown = np.mean(drawdowns)
    return_by_maxdd = net_worth/max_drawdown if max_drawdown!=0 else 0
    return_by_meandd = net_worth/mean_drawdown if mean_drawdown!=0 else 0

    ### Store performance metric information
    pr_results = dict(max_drawdown = max_drawdown,
                      mean_drawdown = mean_drawdown,
                      return_by_maxdd = return_by_maxdd,
                      return_by_meandd = return_by_meandd,
                      net_worth = net_worth
    )

    return pr_results