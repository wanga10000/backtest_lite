import numpy as np
import pandas as pd
from utils.utils_general import normalize_orders_info


def perfmetric_simple(agent_dicts_dict):
    ### Initialize
    position_cnt = 0
    cancel_position_cnt = 0
    order_cnt = 0
    invalid_order_cnt = 0
    win_position_cnt = 0
    loss_position_cnt = 0
    gross_profit = 0
    gross_loss = 0

    ### Accumulate basic metric
    for idx, (agent_name, agent_dict) in enumerate(agent_dicts_dict.items()):
        ### Load order info
        orders_info = agent_dict['orders_info']
        orders_info = normalize_orders_info(orders_info)
        orders_info_noc = orders_info.loc[~(orders_info['cancel']==1)].reset_index(drop = True)
        orders_info_noc_pair = orders_info_noc.groupby('position_cnt').filter(lambda x: len(x) >= 2)

        ### Accumulate invalid order cnt
        invalid_order_cnt += len(orders_info.loc[orders_info['invalid']!='NO'])

        ### Accumulate position cnt
        position_cnt_tmp = len(set(orders_info['position_cnt']))
        position_cnt += position_cnt_tmp
        cancel_position_cnt += len(orders_info.loc[orders_info['base']!=0]) - position_cnt_tmp

        ### Accumulate normalized gross P&L for nor_profit_fac
        orders_info_noc_close_pos = orders_info_noc_pair.loc[orders_info_noc_pair['base']==0].copy()
        win_position_cnt += len(orders_info_noc_close_pos.loc[orders_info_noc_close_pos['pnl'] > 0])
        loss_position_cnt += len(orders_info_noc_close_pos.loc[orders_info_noc_close_pos['pnl'] < 0])
        pos_mask = orders_info_noc_close_pos['pnl'] > 0
        neg_mask = orders_info_noc_close_pos['pnl'] < 0
        gross_profit += (orders_info_noc_close_pos.loc[pos_mask, 'pnl']).sum()
        gross_loss += (-orders_info_noc_close_pos.loc[neg_mask, 'pnl']).sum()

        ### Make eq series.
        eq_series = orders_info_noc_close_pos['pnl'].cumsum() + agent_dict['general']['BALANCE']
        eq_list = [agent_dict['general']['BALANCE']] + eq_series.to_list()
        eq_list += [eq_list[-1]]
        eq_dates = pd.concat([pd.Series(agent_dict['backtest']['START_DATE']), orders_info_noc_close_pos['exec_date']])
        eq_dates = pd.concat([eq_dates, pd.Series([agent_dict['backtest']['STOP_DATE']])])
        eq_series = pd.Series(eq_list, index = eq_dates.values)

        ### Merge eq df
        if idx==0: # First agent_dict
            eq_df = pd.DataFrame(eq_series)
        else:
            temp = pd.merge(eq_df, pd.DataFrame(eq_series), left_index=True, right_index=True, how='outer')
            temp = temp.replace(to_replace=np.nan, method='bfill').replace(to_replace=np.nan, method='ffill')
            eq_df = pd.DataFrame(temp.iloc[:, 0] + temp.iloc[:, 1])

    ### Cancel position percent
    cancel_position_rate = cancel_position_cnt / (cancel_position_cnt + position_cnt) if (cancel_position_cnt + position_cnt) else 0

    ### Invalid group percent
    invalid_order_rate = invalid_order_cnt/order_cnt if order_cnt else 0

    ### Win rate and normalized profit & loss ratio
    win_position_rate = (win_position_cnt / position_cnt) if position_cnt else 0
    nor_pl_ratio = (gross_profit/win_position_cnt) / (gross_loss/loss_position_cnt) if (loss_position_cnt and gross_loss and win_position_cnt) else 0

    ### Sharpe, Sortino, Omega ratio
    eq_day = eq_df.resample('D').last().replace(to_replace=np.nan, method='ffill')
    returns_day = eq_day.pct_change().squeeze()
    pos_return_day = returns_day[returns_day > 0]
    neg_return_day = returns_day[returns_day < 0]
    return_day_mean = returns_day.mean() if len(returns_day) else 0
    return_day_std = returns_day.std() if len(returns_day) else 0
    sharpe = (return_day_mean * np.sqrt(252) / return_day_std) if return_day_std else 0
    if len(neg_return_day)<=1 and len(returns_day)>=1:
        sortino = 10 # Give a big value means infinite
    else:
        neg_return_day_std = neg_return_day.std() if len(neg_return_day) else 0
        sortino = (return_day_mean * np.sqrt(252) / neg_return_day_std) if neg_return_day_std else 0

    omega = pos_return_day.sum() / abs(neg_return_day.sum()) if abs(neg_return_day.sum()) else 0

    ### Normalized Profit factor
    nor_profit_fac = gross_profit / gross_loss if gross_loss else 0

    ### Annaul position count
    total_day = (eq_df.index[-1] - eq_df.index[0]).total_seconds() / (3600*24)
    annual_position_cnt = position_cnt*365/total_day if total_day else 0

    ### drawdowns
    eq_array = eq_df.squeeze().values
    cumulative_max = np.maximum.accumulate(eq_array)
    drawdowns = (cumulative_max - eq_array) / cumulative_max
    max_drawdown = np.max(drawdowns)
    mean_drawdown = np.mean(drawdowns)

    ### Net worth
    net_worth = (eq_array[-1]-eq_array[0])/eq_array[0]

    ### Return by drawdown
    return_by_maxdd = net_worth/max_drawdown if max_drawdown else 0
    return_by_meandd = net_worth/mean_drawdown if mean_drawdown else 0

    ### Store performance metric information
    pr_result = dict(position_cnt = position_cnt,
                     order_cnt = order_cnt,
                     nor_profit = gross_profit,
                     nor_loss = gross_loss,
                     annual_position_cnt = annual_position_cnt,
                     invalid_order_rate = invalid_order_rate,
                     cancel_position_rate = cancel_position_rate,
                     win_position_rate = win_position_rate,
                     nor_pl_ratio = nor_pl_ratio,
                     sharpe = sharpe,
                     sortino = sortino,
                     omega = omega,
                     nor_profit_fac = nor_profit_fac,
                     max_drawdown = max_drawdown,
                     mean_drawdown = mean_drawdown,
                     return_by_maxdd = return_by_maxdd,
                     return_by_meandd = return_by_meandd,
                     net_worth = net_worth,
                     eq_series = eq_df.squeeze()
    )

    return agent_dicts_dict, pr_result
