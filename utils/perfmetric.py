import numpy as np
import pandas as pd


def perfmetric(agent_dicts_dict):
    ### Initialize
    position_cnt = 0
    cancel_position_cnt = 0
    order_cnt = 0
    invalid_order_cnt = 0
    win_position_cnt = 0
    loss_position_cnt = 0
    normalized_gross_profit = 0
    normalized_gross_loss = 0
    normalized_gross_profit_long = 0
    normalized_gross_loss_long = 0
    total_money_flow = 0
    trade_fee_sum = 0
    spread_fee_sum = 0
    slippage_fee_sum = 0
    swap_fee_sum = 0
    rmul_df = pd.DataFrame()
    hold_position_hour = 0
    nor_mfe_sum = 0
    nor_mae_sum = 0
    pnl_in_mfae_list = []

    ### Accumulate basic metric
    for idx, (agent_name, agent_dict) in enumerate(agent_dicts_dict.items()):
        ### Load order info
        orders_info = agent_dict['orders_info']
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
        orders_info_noc_close_pos['margin'] = abs(orders_info_noc_pair.iloc[1::2]['exec_base'].values) * orders_info_noc_pair.iloc[::2]['exec_price'].values
        if len(orders_info_noc_close_pos)!=0:
            initial_margin = abs(orders_info_noc_close_pos['margin'].iloc[0])
        else:
            initial_margin = 0
        orders_info_noc_close_pos['nor_pnl'] = orders_info_noc_close_pos['pnl']*initial_margin / orders_info_noc_close_pos['margin']
        pos_mask = orders_info_noc_close_pos['nor_pnl'] > 0
        neg_mask = orders_info_noc_close_pos['nor_pnl'] < 0
        normalized_gross_profit += (orders_info_noc_close_pos.loc[pos_mask, 'nor_pnl']).sum()
        normalized_gross_loss += (-orders_info_noc_close_pos.loc[neg_mask, 'nor_pnl']).sum()

        ### Accumulate normalized gross P&L for long short compare
        long_pos_mask = (orders_info_noc_close_pos['exec_base'] < 0) & pos_mask
        long_neg_mask = (orders_info_noc_close_pos['exec_base'] < 0) & neg_mask
        normalized_gross_profit_long += (orders_info_noc_close_pos.loc[long_pos_mask, 'nor_pnl']).sum()
        normalized_gross_loss_long += (-orders_info_noc_close_pos.loc[long_neg_mask, 'nor_pnl']).sum()

        ### Other statistic
        order_cnt += len(orders_info)
        trade_fee_sum += orders_info_noc['trade_fee'].sum()
        spread_fee_sum += orders_info_noc['spread_fee'].sum()
        slippage_fee_sum += orders_info_noc['slippage_fee'].sum()
        swap_fee_sum += orders_info_noc['swap_fee'].sum()
        total_money_flow += orders_info_noc_close_pos['pnl'].abs().sum()

        ### Accumulate hold position hour
        if len(orders_info_noc_pair)!=0:
            total_second = orders_info_noc_pair.groupby('position_cnt').agg({'exec_date': lambda x: (x.iloc[1] - x.iloc[0]).total_seconds()}).squeeze().sum()
        else:
            total_second = 0
        hold_position_hour += total_second / 3600

        ### Accumulate risk stuff using 1% risk
        rmul_df_tmp = orders_info_noc_close_pos['pnl'] / (orders_info_noc_close_pos['margin'] * 0.01)
        rmul_df_tmp = rmul_df_tmp.fillna(0)
        rmul_df = pd.concat([rmul_df, rmul_df_tmp])

        ### Accumulate edge ratio stuff
        nor_mfe_sum += (orders_info_noc_close_pos['mfe']*initial_margin / orders_info_noc_close_pos['margin']).sum()
        nor_mae_sum += (orders_info_noc_close_pos['mae']*initial_margin / orders_info_noc_close_pos['margin']).sum()
        pnl_in_mfae_tmp = (orders_info_noc_close_pos['pnl'] - (-orders_info_noc_close_pos['mae'])) / (orders_info_noc_close_pos['mfe'] - (-orders_info_noc_close_pos['mae']))
        pnl_in_mfae_list += pnl_in_mfae_tmp.to_list()

        ### Get eq series. If agent_dict doesn't have it, make a simple one.
        has_eq = 0
        if 'eq_series' in agent_dict.keys():
            if len(agent_dict['eq_series'])!=0:
                eq_series = agent_dict['eq_series']
                has_eq = 1

        if has_eq==0:
            eq_series = orders_info_noc_close_pos['pnl'].cumsum() + agent_dict['general']['BALANCE']
            eq_list = [agent_dict['general']['BALANCE']] + eq_series.to_list()
            eq_list += [eq_list[-1]]
            eq_dates = pd.concat([pd.Series(agent_dict['backtest']['START_DATE']), orders_info_noc_close_pos['exec_date']])
            eq_dates = pd.concat([eq_dates, pd.Series([agent_dict['backtest']['STOP_DATE']])])
            eq_series = pd.Series(eq_list, index = eq_dates.values)
            agent_dicts_dict[agent_name]['eq_series'] = eq_series

        ### Merge eq df
        if idx==0: # First agent_dict
            eq_df = pd.DataFrame(eq_series)
        else:
            temp = pd.merge(eq_df, pd.DataFrame(eq_series), left_index=True, right_index=True, how='outer')
            temp = temp.replace(to_replace=np.nan, method='bfill').replace(to_replace=np.nan, method='ffill')
            eq_df = pd.DataFrame(temp.iloc[:, 0] + temp.iloc[:, 1])

        ### Make normalized eq series and merge
        nor_eq_series = orders_info_noc_close_pos['nor_pnl'].cumsum() + agent_dict['general']['BALANCE']
        nor_eq_list = [agent_dict['general']['BALANCE']] + nor_eq_series.to_list()
        nor_eq_list += [nor_eq_list[-1]]
        nor_eq_dates = pd.concat([pd.Series(agent_dict['backtest']['START_DATE']), orders_info_noc_close_pos['exec_date']])
        nor_eq_dates = pd.concat([nor_eq_dates, pd.Series([agent_dict['backtest']['STOP_DATE']])])
        nor_eq_series = pd.Series(nor_eq_list, index = nor_eq_dates.values)


        if idx==0: # First agent_dict
            nor_eq_df = pd.DataFrame(nor_eq_series)
        else:
            temp = pd.merge(nor_eq_df, pd.DataFrame(nor_eq_series), left_index=True, right_index=True, how='outer')
            temp = temp.replace(to_replace=np.nan, method='bfill').replace(to_replace=np.nan, method='ffill')
            nor_eq_df = pd.DataFrame(temp.iloc[:, 0] + temp.iloc[:, 1])

    ### Cancel position percent
    cancel_position_rate = cancel_position_cnt / (cancel_position_cnt + position_cnt) if (cancel_position_cnt + position_cnt) else 0

    ### Invalid group percent
    invalid_order_rate = invalid_order_cnt/order_cnt if order_cnt else 0

    ### Win rate and normalized profit & loss ratio
    win_position_rate = (win_position_cnt / position_cnt) if position_cnt else 0
    nor_pl_ratio = (normalized_gross_profit/win_position_cnt) / (normalized_gross_loss/loss_position_cnt) if (loss_position_cnt and normalized_gross_loss and win_position_cnt) else 0

    ### Average fee percent
    avg_trade_fee_rate = trade_fee_sum/total_money_flow if total_money_flow else 0
    avg_spread_fee_rate = spread_fee_sum/total_money_flow if total_money_flow else 0
    avg_slippage_fee_rate = slippage_fee_sum/total_money_flow if total_money_flow else 0
    avg_swap_fee_rate = swap_fee_sum/total_money_flow if total_money_flow else 0

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

    ### Sharpe month, Sortino month, Omega month ratio
    eq_mon = eq_df.resample('M').last().replace(to_replace=np.nan, method='ffill')
    returns_mon = eq_mon.pct_change().squeeze()
    return_mon_mean = returns_mon.mean()
    return_mon_std = returns_mon.std()
    sharpe_mon = (return_mon_mean * np.sqrt(12) / return_mon_std) if return_mon_std else 0

    ### Normalized Profit factor
    nor_profit_fac = normalized_gross_profit / normalized_gross_loss if normalized_gross_loss else 0

    ### Annaul position count
    total_day = (eq_df.index[-1] - eq_df.index[0]).total_seconds() / (3600*24)
    annual_position_cnt = position_cnt*365/total_day if total_day else 0

    ### Statistical significant edge ratio
    ss_win_cnt = 0
    SS_EDGE_CNT = 10000
    for _ in range(SS_EDGE_CNT):
        outcomes = np.random.rand(int(annual_position_cnt)) < win_position_rate
        result = np.where(outcomes, nor_pl_ratio, -1)
        if result.sum()>0:
            ss_win_cnt += 1
    ss_edge = ss_win_cnt/SS_EDGE_CNT

    ### Van Tharp's
    expectancy = rmul_df.sum()[0]/position_cnt if position_cnt else 0
    rmul_array = rmul_df.values
    rmul_std = np.std(rmul_array) if len(rmul_array)!=0 else 0
    annual_position_cnt_per_agent = annual_position_cnt / len(agent_dicts_dict)
    position_cnt_thd = 500 if (annual_position_cnt_per_agent>500) else annual_position_cnt_per_agent
    sqn = (expectancy/rmul_std)*(np.sqrt(position_cnt_thd)) if rmul_std else 0
    sqn_by_sharpe = sharpe * (np.sqrt(position_cnt_thd))
    sqn_by_sortino = sortino * (np.sqrt(position_cnt_thd))

    ### Edge ratio
    nor_edge_ratio = (nor_mfe_sum / nor_mae_sum) if nor_mae_sum else 0
    exit_efficiency = np.mean(pnl_in_mfae_list) if pnl_in_mfae_list else 0

    ### drawdowns
    eq_array = eq_df.squeeze().values
    cumulative_max = np.maximum.accumulate(eq_array)
    drawdowns = (cumulative_max - eq_array) / cumulative_max
    max_drawdown = np.max(drawdowns)
    mean_drawdown = np.mean(drawdowns)

    ### Net worth
    net_worth = (eq_array[-1]-eq_array[0])/eq_array[0]

    ### long_short_percent
    long_profit_percent = normalized_gross_profit_long/normalized_gross_profit if normalized_gross_profit else 0
    long_loss_percent = normalized_gross_loss_long/normalized_gross_loss if normalized_gross_loss else 0

    ### CAGR
    cagr = (eq_array[-1]/eq_array[0]) ** (365/total_day) - 1 if (total_day and eq_array[-1]>0) else 0

    ### Return by drawdown
    return_by_maxdd = net_worth/max_drawdown if max_drawdown else 0
    return_by_meandd = net_worth/mean_drawdown if mean_drawdown else 0

    ### Coefficient of correlation (nor because it use fixed pos)
    nor_eq_series = nor_eq_df.resample('D').last().replace(to_replace=np.nan, method='ffill').squeeze()
    nor_eq_array = nor_eq_series.values
    nor_coef_r = np.corrcoef(range(len(nor_eq_array)), nor_eq_array)[0, 1] if not len(set(nor_eq_array)) == 1 else 0

    ### Avg trade
    nor_avg_trade = (nor_eq_array [-1]-nor_eq_array [0])/position_cnt if position_cnt else 0

    ### Avg hold position hour
    avg_hold_position_hour = hold_position_hour / position_cnt if position_cnt else 0

    ### Custom metric (not print)
    nor_profit_fac_ss_edge = nor_profit_fac * ((ss_edge + 1)/2)

    ### Store performance metric information
    pr_result = dict(position_cnt = position_cnt,
                     order_cnt = order_cnt,
                     nor_profit = normalized_gross_profit,
                     nor_loss = normalized_gross_loss,
                     annual_position_cnt = annual_position_cnt,
                     invalid_order_rate = invalid_order_rate,
                     cancel_position_rate = cancel_position_rate,
                     win_position_rate = win_position_rate,
                     nor_pl_ratio = nor_pl_ratio,
                     ss_edge = ss_edge,
                     nor_avg_trade = nor_avg_trade,
                     avg_hold_position_hour = avg_hold_position_hour,
                     avg_trade_fee_rate = avg_trade_fee_rate,
                     avg_spread_fee_rate = avg_spread_fee_rate,
                     avg_slippage_fee_rate = avg_slippage_fee_rate,
                     avg_swap_fee_rate = avg_swap_fee_rate,
                     sharpe = sharpe,
                     sharpe_mon = sharpe_mon,
                     sortino = sortino,
                     omega = omega,
                     nor_profit_fac = nor_profit_fac,
                     nor_coef_r = nor_coef_r,
                     sqn = sqn,
                     sqn_by_sharpe = sqn_by_sharpe,
                     sqn_by_sortino = sqn_by_sortino,
                     nor_edge_ratio = nor_edge_ratio,
                     exit_efficiency = exit_efficiency,
                     max_drawdown = max_drawdown,
                     mean_drawdown = mean_drawdown,
                     cagr = cagr,
                     return_by_maxdd = return_by_maxdd,
                     return_by_meandd = return_by_meandd,
                     long_profit_percent = long_profit_percent,
                     long_loss_percent = long_loss_percent,
                     net_worth = net_worth,
                     nor_profit_fac_ss_edge = nor_profit_fac_ss_edge,
                     eq_series = eq_df.squeeze()
    )

    return agent_dicts_dict, pr_result
