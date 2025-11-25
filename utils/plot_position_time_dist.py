import matplotlib.pyplot as plt
import numpy as np
from utils.ind_histogram import ind_histogram


def plot_position_time_dist(agent_dicts_dict, ind_type, pend_exec, entry_exit, MULTI_PROCESSING):
    # ind_type: 'hour', 'weekday'
    # pend_exec: 0 (pending_date) 1 (exec_date)
    # entry_exit: 0 (entry) 1 (exit)

    # Specify ind range
    ind_min = 0
    ind_max = 24 if ind_type=='hour' else 7
    ind_bin = 24 if ind_type=='hour' else 7
    ind_range = np.arange(ind_min, ind_max, (ind_max-ind_min)/ind_bin)

    # Add ind value for ind_histogram
    for agent_name, agent_dict in agent_dicts_dict.items():
        if ind_type=='hour':
            agent_dict['orders_info']['pending_ind'] = agent_dict['orders_info']['pending_date'].dt.hour
            agent_dict['orders_info']['exec_ind'] = agent_dict['orders_info']['exec_date'].dt.hour
        else:
            agent_dict['orders_info']['pending_ind'] = agent_dict['orders_info']['pending_date'].dt.weekday
            agent_dict['orders_info']['exec_ind'] = agent_dict['orders_info']['exec_date'].dt.weekday
        agent_dicts_dict[agent_name] = agent_dict

    # Run ind analysis
    pr_result_hist = ind_histogram(agent_dicts_dict, ind_range, pend_exec, entry_exit, MULTI_PROCESSING, print_progress=0)

    # Plot annual_position_cnt / nor gross win/loss/diff
    plt.ioff()
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    plt.rcParams['font.size'] = '12'

    axs[0].bar(ind_range, pr_result_hist['annual_position_cnt'],
            width=(ind_max - ind_min) / (ind_bin * 2))
    axs[0].set_ylabel('annual_position_cnt')
    axs[0].set_title('Annual Position Count')
    profits = pr_result_hist['nor_profit']
    losses = pr_result_hist['nor_loss']
    profit_loss = profits - losses

    bar_width = (ind_max - ind_min) / (ind_bin * 2 * 3)
    x = np.array(ind_range)
    axs[1].bar(x - bar_width, profits, width=bar_width, label='Profit')
    axs[1].bar(x, losses, width=bar_width, label='Loss')
    axs[1].bar(x + bar_width, profit_loss, width=bar_width, label='Profit - Loss')
    axs[1].set_xlabel(ind_type)
    axs[1].set_xticks(ind_range)
    axs[1].legend()

    plt.tight_layout()

    return fig, axs