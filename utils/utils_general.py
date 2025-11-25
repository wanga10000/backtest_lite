import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mpl_dates
from datetime import datetime, timedelta
from ta.volatility import AverageTrueRange
import math
import re
import os
import glob
import stat, shutil
from copy import deepcopy


def read_df_sets(agent_dict):
    ### Initialize from agent_dict
    START_DATE = agent_dict['backtest']['START_DATE']
    STOP_DATE = agent_dict['backtest']['STOP_DATE']
    SYMBOLS = str(agent_dict['general']['SYMBOLS']).split(",")
    USED_TFS = str(agent_dict['general']['USED_TFS']).split(",")
    TF_OFFSET = str(agent_dict['general']['TF_OFFSET']).split(",")
    BUFSIZE = list(map(int, str(agent_dict['general']['BUFSIZE']).split(",")))
    NO_COST = agent_dict['backtest']['NO_COST']

    ### Pre define
    tf_pd_map = {'0':'0', 'm':'T', 'h':'H', 'd':'D', 'w':'W', 'n':'M'}
    unit_minute_map = {'m':1, 'h':60, 'd':1440}

    ### Symbol for loop
    df_sets = []
    for sidx, _ in enumerate(SYMBOLS):
        symbol_info = agent_dict['symbols_info'][sidx]

        ### Get valid tf
        tf_valid_key = glob.glob(f"{symbol_info['DATA_PATH'] + '/'}"+'[0-9]*.parquet')
        tf_valid_key = [os.path.splitext(os.path.basename(key))[0] for key in tf_valid_key]
        tf_valid_key = [key for key in tf_valid_key if key[-1]!='n']
        tf_valid_minute_map = {int(tf[:-1]) * unit_minute_map[tf[-1]] : tf for tf in tf_valid_key}
        tf_valid_minute_map = dict(sorted(tf_valid_minute_map.items()))

        ### Get df_set
        ### Basic logic behind the codes:
        # Loop USED_TFS by tf
        # Check if required sampled by smaller tf, then resample + offset by pd
        # if not, than get df without resample + offset
        df_set = []
        for idx, tf in enumerate(USED_TFS):
            used_tf_minute = int(tf[:-1]) * unit_minute_map[tf[-1]]
            offset_minute = int(TF_OFFSET[idx][:-1]) * unit_minute_map[TF_OFFSET[idx][-1]] if '0' not in TF_OFFSET[idx] else used_tf_minute
            for used_tf_minute_api_tf in reversed(tf_valid_minute_map.keys()):
                if used_tf_minute_api_tf==math.gcd(used_tf_minute_api_tf, used_tf_minute):
                    break
            for offset_minute_api_tf in reversed(tf_valid_minute_map.keys()):
                if offset_minute_api_tf==math.gcd(offset_minute_api_tf, offset_minute):
                    break
            tf_minute = min(used_tf_minute_api_tf, offset_minute_api_tf)
            tf_min = tf_valid_minute_map[tf_minute]

            ### Load df min and process spread
            df_min = pd.read_parquet(symbol_info['DATA_PATH'] + '/' + tf_min + '.parquet')
            df_min = df_min.set_index('Date')

            ### Process spread
            SPREAD_POS = int(symbol_info['SPREAD_POS'])
            SPREAD_FACTOR = float(symbol_info['SPREAD_FACTOR'])
            DEFAULT_SPREAD = float(symbol_info['DEFAULT_SPREAD'])

            df_min['Spread'] = df_min['Spread']*SPREAD_FACTOR if 'Spread' in df_min else DEFAULT_SPREAD
            df_min['Spread'] = 0 if NO_COST==1 else df_min['Spread']
            df_min['Open_ask'] = df_min['Open'] + df_min['Spread'] * (1/(10**SPREAD_POS))
            df_min['High_ask'] = df_min['High'] + df_min['Spread'] * (1/(10**SPREAD_POS))
            df_min['Low_ask'] = df_min['Low'] + df_min['Spread'] * (1/(10**SPREAD_POS))

            ### Check if required sampled by smaller tf
            if tf_min==tf:
                df = df_min.copy(deep=True)

            else:
                pd_tf = tf[:-1] + tf_pd_map[tf[-1]]
                pd_tf_offset = TF_OFFSET[idx][:-1] + tf_pd_map[TF_OFFSET[idx][-1]]
                df = df_min.resample(rule=pd_tf, offset=pd_tf_offset, origin='start_day').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum',
                    'Spread': 'first',
                    'Open_ask': 'first',
                    'High_ask': 'max',
                    'Low_ask': 'min'
                })
                df.dropna(inplace=True)

            df_date = df.index.values.squeeze()
            start_idx = get_rightmost_time_idx(np.datetime64(START_DATE), df_date) - BUFSIZE[idx]
            stop_idx = get_leftmost_time_idx(np.datetime64(STOP_DATE), df_date) + 1
            start_idx = 0 if start_idx<0 else start_idx
            df = df.iloc[start_idx:stop_idx, :]
            df_set.append(df)

        df_sets.append(df_set)

    return df_sets


def read_df_sets_add_atr_noise(agent_dict, gen_num=1, atr_noise=0.1):
    ### Initialize from agent_dict
    START_DATE = agent_dict['backtest']['START_DATE']
    STOP_DATE = agent_dict['backtest']['STOP_DATE']
    SYMBOLS = str(agent_dict['general']['SYMBOLS']).split(",")
    USED_TFS = str(agent_dict['general']['USED_TFS']).split(",")
    TF_OFFSET = str(agent_dict['general']['TF_OFFSET']).split(",")
    BUFSIZE = list(map(int, str(agent_dict['general']['BUFSIZE']).split(",")))
    NO_COST = agent_dict['backtest']['NO_COST']
    REAL_TICK = agent_dict['backtest']['REAL_TICK']
    FAKE_TICK_TFS = str(agent_dict['backtest']['FAKE_TICK_TF']).split(",")
    if len(FAKE_TICK_TFS) == 1: # If only one tf is provided, use it for all symbols
        FAKE_TICK_TFS = FAKE_TICK_TFS * len(SYMBOLS)

    ### Pre define
    tf_pd_map = {'0':'0', 'm':'T', 'h':'H', 'd':'D', 'w':'W', 'n':'M'}
    unit_minute_map = {'m':1, 'h':60, 'd':1440}

    ### Structure declare
    df_sets_list = [[0] * len(SYMBOLS) for _ in range(int(gen_num))]
    tick_df_sets_list = [[0] * len(SYMBOLS) for _ in range(int(gen_num))]

    ### Symbol for loop
    for sidx, _ in enumerate(SYMBOLS):
        symbol_info = agent_dict['symbols_info'][sidx]

        ### Get min df
        tf_valid_key = glob.glob(f"{symbol_info['DATA_PATH'] + '/'}"+'[0-9]*.parquet')
        tf_valid_key = [os.path.splitext(os.path.basename(key))[0] for key in tf_valid_key]
        tf_valid_key = [key for key in tf_valid_key if key[-1]!='n']
        tf_valid_minute_map = {int(tf[:-1]) * unit_minute_map[tf[-1]] : tf for tf in tf_valid_key}
        tf_valid_minute_map = dict(sorted(tf_valid_minute_map.items()))
        tf_min = next(iter(tf_valid_minute_map.values()))
        df_min = pd.read_parquet(symbol_info['DATA_PATH'] + '/' + tf_min + '.parquet')
        df_min = df_min.set_index('Date')

        ### Add ATR noise on min df
        # Clip to required date to save ATR calculate time
        unit_minute_map = {'m':1, 'h':60, 'd':1440}
        min_tf = USED_TFS[0]
        min_tf_min = int(min_tf[:-1]) * unit_minute_map[min_tf[-1]]
        max_bufsize = 0
        for ti, tf in enumerate(USED_TFS):
            used_tf_min = int(tf[:-1]) * unit_minute_map[tf[-1]]
            bufsize_min = int(BUFSIZE[ti]*(used_tf_min / min_tf_min)+5)
            max_bufsize = bufsize_min if bufsize_min>max_bufsize else max_bufsize

        start_idx = get_rightmost_time_idx(np.datetime64(START_DATE), df_min.index) - max_bufsize
        stop_idx = get_leftmost_time_idx(np.datetime64(STOP_DATE), df_min.index) + 1
        start_idx = 0 if start_idx<0 else start_idx
        df_min = df_min.iloc[start_idx:stop_idx, :]

        # Add noise
        df_min_atr = AverageTrueRange(high=df_min['High'],
                                      low=df_min['Low'],
                                      close=df_min['Close'],
                                      window=14).average_true_range()

        ### Loop for gen_num of gen
        for gidx in range(int(gen_num)):
            df_min_copy = deepcopy(df_min)
            df_min_noise = np.random.uniform(-atr_noise, atr_noise, size=len(df_min_copy['Close']))
            df_min_noise = df_min_atr.values * df_min_noise

            # HL first
            df_min_copy['High'] = add_noise_on_delta(df_min_copy['High'], df_min_noise)
            df_min_copy['Low'] = add_noise_on_delta(df_min_copy['Low'], df_min_noise)

            # OC need to check HL
            close_tmp = add_noise_on_delta(df_min_copy['Close'], df_min_noise)
            df_min_copy['Close'] = close_tmp.clip(lower=df_min_copy['Low'], upper=df_min_copy['High'])
            open_tmp = add_noise_on_delta(df_min_copy['Open'], df_min_noise)
            df_min_copy['Open'] = open_tmp.clip(lower=df_min_copy['Low'], upper=df_min_copy['High'])

            # If some price smaller than zero, add an offset to make all price>0.1 mean
            min_value = df_min_copy.iloc[:, 0:4].min().min()
            if min_value < 0:
                offset = 0.1*df_min_copy['Close'].mean() - min_value
                df_min_copy.iloc[:, 0:4] = df_min_copy.iloc[:, 0:4] + offset

            # Process spread
            SPREAD_POS = int(symbol_info['SPREAD_POS'])
            SPREAD_FACTOR = float(symbol_info['SPREAD_FACTOR'])
            DEFAULT_SPREAD = float(symbol_info['DEFAULT_SPREAD'])

            df_min_copy['Spread'] = df_min_copy['Spread']*SPREAD_FACTOR if 'Spread' in df_min_copy else DEFAULT_SPREAD
            df_min_copy['Spread'] = 0 if NO_COST==1 else df_min_copy['Spread']
            df_min_copy['Open_ask'] = df_min_copy['Open'] + df_min_copy['Spread'] * (1/(10**SPREAD_POS))
            df_min_copy['High_ask'] = df_min_copy['High'] + df_min_copy['Spread'] * (1/(10**SPREAD_POS))
            df_min_copy['Low_ask'] = df_min_copy['Low'] + df_min_copy['Spread'] * (1/(10**SPREAD_POS))

            ### Get df_set by min tf
            df_set = []
            for idx, tf in enumerate(USED_TFS):
                ### Check if required sampled by smaller tf
                if tf_min==tf:
                    df = df_min_copy
                else:
                    pd_tf = tf[:-1] + tf_pd_map[tf[-1]]
                    pd_tf_offset = TF_OFFSET[idx][:-1] + tf_pd_map[TF_OFFSET[idx][-1]]
                    df = df_min_copy.resample(rule=pd_tf, offset=pd_tf_offset, origin='start_day').agg({
                        'Open': 'first',
                        'High': 'max',
                        'Low': 'min',
                        'Close': 'last',
                        'Volume': 'sum',
                        'Spread': 'first',
                        'Open_ask': 'first',
                        'High_ask': 'max',
                        'Low_ask': 'min'
                    })
                    df.dropna(inplace=True)

                df_date = df.index.values.squeeze()
                start_idx = get_rightmost_time_idx(np.datetime64(START_DATE), df_date) - BUFSIZE[idx]
                stop_idx = get_leftmost_time_idx(np.datetime64(STOP_DATE), df_date) + 1
                start_idx = 0 if start_idx<0 else start_idx
                df = df.iloc[start_idx:stop_idx, :]
                df_set.append(df)

            df_sets_list[gidx][sidx] = df_set

            # After df_set done, make fake df, remove spread & volumn & ask from df_min_copy and add to tick_df_sets
            if REAL_TICK==0:
                fake_tick_tf = FAKE_TICK_TFS[sidx]
                df_min_copy2 = df_min_copy.copy(deep=True)
                df_min_copy2 = df_min_copy2.iloc[:, :4]
                pd_tf = fake_tick_tf[:-1] + tf_pd_map[fake_tick_tf[-1]]
                tick_df = df_min_copy2.resample(rule=pd_tf, offset='0', origin='start_day').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                })
                tick_df.dropna(inplace=True)
                tick_df = tick_df.reset_index().rename(columns={'index': 'Date'})
                tick_df_sets_list[gidx][sidx] = tick_df

    # Final return: list of dict, list stands for gen_num of gen, dict has 2 keys: df_set and tick_df_set
    df_dicts_list = [{"df_sets": df_sets_list[gidx], "tick_df_sets": tick_df_sets_list[gidx]} for gidx in range(int(gen_num))]

    return df_dicts_list


class atr_noise_df_sets_generator():
    def __init__(self, agent_dict):
        ### Initialize from agent_dict
        self.START_DATE = agent_dict['backtest']['START_DATE']
        self.STOP_DATE = agent_dict['backtest']['STOP_DATE']
        self.SYMBOLS = str(agent_dict['general']['SYMBOLS']).split(",")
        self.symbols_info = agent_dict['symbols_info']
        self.USED_TFS = str(agent_dict['general']['USED_TFS']).split(",")
        self.TF_OFFSET = str(agent_dict['general']['TF_OFFSET']).split(",")
        self.BUFSIZE = list(map(int, str(agent_dict['general']['BUFSIZE']).split(",")))
        self.NO_COST = agent_dict['backtest']['NO_COST']
        self.REAL_TICK = agent_dict['backtest']['REAL_TICK']
        self.FAKE_TICK_TFS = str(agent_dict['backtest']['FAKE_TICK_TF']).split(",")
        if len(self.FAKE_TICK_TFS) == 1: # If only one tf is provided, use it for all symbols
            self.FAKE_TICK_TFS = self.FAKE_TICK_TFS * len(self.SYMBOLS)

        ### Pre define
        self.tf_pd_map = {'0':'0', 'm':'T', 'h':'H', 'd':'D', 'w':'W', 'n':'M'}
        self.unit_minute_map = {'m':1, 'h':60, 'd':1440}

        ### Make df_min for symbols
        self.df_min_list = []
        self.df_min_atrs_list = []
        self.tf_min_list = []

        for sidx, _ in enumerate(self.SYMBOLS):
            symbol_info = self.symbols_info[sidx]

            ### Get min df
            tf_valid_key = glob.glob(f"{symbol_info['DATA_PATH'] + '/'}"+'[0-9]*.parquet')
            tf_valid_key = [os.path.splitext(os.path.basename(key))[0] for key in tf_valid_key]
            tf_valid_key = [key for key in tf_valid_key if key[-1]!='n']
            tf_valid_minute_map = {int(tf[:-1]) * self.unit_minute_map[tf[-1]] : tf for tf in tf_valid_key}
            tf_valid_minute_map = dict(sorted(tf_valid_minute_map.items()))
            tf_min = next(iter(tf_valid_minute_map.values()))
            df_min = pd.read_parquet(symbol_info['DATA_PATH'] + '/' + tf_min + '.parquet')
            df_min = df_min.set_index('Date')

            ### Add ATR noise on min df
            # Clip to required date to save ATR calculate time
            min_tf = self.USED_TFS[0]
            min_tf_min = int(min_tf[:-1]) * self.unit_minute_map[min_tf[-1]]
            max_bufsize = 0
            for ti, tf in enumerate(self.USED_TFS):
                used_tf_min = int(tf[:-1]) * self.unit_minute_map[tf[-1]]
                bufsize_min = int(self.BUFSIZE[ti]*(used_tf_min / min_tf_min)+5)
                max_bufsize = bufsize_min if bufsize_min>max_bufsize else max_bufsize

            start_idx = get_rightmost_time_idx(np.datetime64(self.START_DATE), df_min.index) - max_bufsize
            stop_idx = get_leftmost_time_idx(np.datetime64(self.STOP_DATE), df_min.index) + 1
            start_idx = 0 if start_idx<0 else start_idx
            df_min = df_min.iloc[start_idx:stop_idx, :]

            # Add noise
            df_min_atr = AverageTrueRange(high=df_min['High'],
                                          low=df_min['Low'],
                                          close=df_min['Close'],
                                          window=14).average_true_range()

            self.df_min_list.append(df_min)
            self.df_min_atrs_list.append(df_min_atr)
            self.tf_min_list.append(tf_min)


    def gen_1_set(self, atr_noise):
        df_sets = []
        tick_df_sets = []

        ### Loop for symbols
        for sidx, _ in enumerate(self.SYMBOLS):
            df_min_copy = deepcopy(self.df_min_list[sidx])
            df_min_noise = np.random.uniform(-atr_noise, atr_noise, size=len(df_min_copy['Close']))
            df_min_noise = self.df_min_atrs_list[sidx].values * df_min_noise

            # HL first
            df_min_copy['High'] = add_noise_on_delta(df_min_copy['High'], df_min_noise)
            df_min_copy['Low'] = add_noise_on_delta(df_min_copy['Low'], df_min_noise)

            # OC need to check HL
            close_tmp = add_noise_on_delta(df_min_copy['Close'], df_min_noise)
            df_min_copy['Close'] = close_tmp.clip(lower=df_min_copy['Low'], upper=df_min_copy['High'])
            open_tmp = add_noise_on_delta(df_min_copy['Open'], df_min_noise)
            df_min_copy['Open'] = open_tmp.clip(lower=df_min_copy['Low'], upper=df_min_copy['High'])

            # If some price smaller than zero, add an offset to make all price>0.1 mean
            min_value = df_min_copy.iloc[:, 0:4].min().min()
            if min_value < 0:
                offset = 0.1*df_min_copy['Close'].mean() - min_value
                df_min_copy.iloc[:, 0:4] = df_min_copy.iloc[:, 0:4] + offset

            # Process spread
            SPREAD_POS = int(self.symbols_info[sidx]['SPREAD_POS'])
            SPREAD_FACTOR = float(self.symbols_info[sidx]['SPREAD_FACTOR'])
            DEFAULT_SPREAD = float(self.symbols_info[sidx]['DEFAULT_SPREAD'])

            df_min_copy['Spread'] = df_min_copy['Spread']*SPREAD_FACTOR if 'Spread' in df_min_copy else DEFAULT_SPREAD
            df_min_copy['Spread'] = 0 if self.NO_COST==1 else df_min_copy['Spread']
            df_min_copy['Open_ask'] = df_min_copy['Open'] + df_min_copy['Spread'] * (1/(10**SPREAD_POS))
            df_min_copy['High_ask'] = df_min_copy['High'] + df_min_copy['Spread'] * (1/(10**SPREAD_POS))
            df_min_copy['Low_ask'] = df_min_copy['Low'] + df_min_copy['Spread'] * (1/(10**SPREAD_POS))

            ### Get df_set by min tf
            df_set = []
            for idx, tf in enumerate(self.USED_TFS):
                ### Check if required sampled by smaller tf
                if self.tf_min_list[sidx]==tf:
                    df = df_min_copy
                else:
                    pd_tf = tf[:-1] + self.tf_pd_map[tf[-1]]
                    pd_tf_offset = self.TF_OFFSET[idx][:-1] + self.tf_pd_map[self.TF_OFFSET[idx][-1]]
                    df = df_min_copy.resample(rule=pd_tf, offset=pd_tf_offset, origin='start_day').agg({
                        'Open': 'first',
                        'High': 'max',
                        'Low': 'min',
                        'Close': 'last',
                        'Volume': 'sum',
                        'Spread': 'first',
                        'Open_ask': 'first',
                        'High_ask': 'max',
                        'Low_ask': 'min'
                    })
                    df.dropna(inplace=True)

                df_date = df.index.values.squeeze()
                start_idx = get_rightmost_time_idx(np.datetime64(self.START_DATE), df_date) - self.BUFSIZE[idx]
                stop_idx = get_leftmost_time_idx(np.datetime64(self.STOP_DATE), df_date) + 1
                start_idx = 0 if start_idx<0 else start_idx
                df = df.iloc[start_idx:stop_idx, :]
                df_set.append(df)

            df_sets.append(df_set)

            # After df_set done, make fake df, remove spread & volumn & ask from df_min_copy and add to tick_df_sets
            if self.REAL_TICK==0:
                fake_tick_tf = self.FAKE_TICK_TFS[sidx]
                df_min_copy2 = df_min_copy.copy(deep=True)
                df_min_copy2 = df_min_copy2.iloc[:, :4]
                pd_tf = fake_tick_tf[:-1] + self.tf_pd_map[fake_tick_tf[-1]]
                tick_df = df_min_copy2.resample(rule=pd_tf, offset='0', origin='start_day').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                })
                tick_df.dropna(inplace=True)
                tick_df = tick_df.reset_index().rename(columns={'index': 'Date'})
                tick_df_sets.append(tick_df)

        # Final return: dict has 2 keys: df_set and tick_df_set
        atr_noise_df_dict = {"df_sets": df_sets, "tick_df_sets": tick_df_sets}

        return atr_noise_df_dict


def add_noise_on_delta(price, atr_noise):
    price_delta = price.diff(1)
    price_delta = price_delta + atr_noise
    price_delta.iloc[0] = price.iloc[0]
    new_price = price_delta.cumsum()
    return new_price


def get_leftmost_time_idx(target_time, date_series):
    idx = np.searchsorted(date_series, target_time, side="right")-1
    return idx


def get_rightmost_time_idx(target_time, date_series):
    idx = np.searchsorted(date_series, target_time, side="left")
    return idx


def polyfit(x, y, degree):
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    #calculate r-squared
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y - ybar)**2)
    r_squared = ssreg / sstot

    return coeffs, r_squared


def report_pr(pr_results, result_txt_path=None, result_png_path=None, print_on=1):
    text = (
        f"Position count: {pr_results['position_cnt']}\n"
        f"Order count: {pr_results['order_cnt']}\n"
        f"Annual position count: {pr_results['annual_position_cnt']:.6f}\n"
        f"Invalid order rate: {pr_results['invalid_order_rate'] * 100:.4f}%\n"
        f"Cancel position rate: {pr_results['cancel_position_rate'] * 100:.4f}%\n"
        f"Win position rate: {pr_results['win_position_rate'] * 100:.4f}%\n"
        f"Profit and loss ratio: {pr_results['nor_pl_ratio']:.4f}\n"
        f"Statistical Significant edge: {pr_results['ss_edge']:.4f}\n"
        f"Average trade: {pr_results['nor_avg_trade']:.6f}\n"
        f"Average hold position hour: {pr_results['avg_hold_position_hour']:.6f}\n"
        f"Average trade fee percent: {pr_results['avg_trade_fee_rate'] * 100:.4f}%\n"
        f"Average spread fee percent: {pr_results['avg_spread_fee_rate'] * 100:.4f}%\n"
        f"Average slippage fee percent: {pr_results['avg_slippage_fee_rate'] * 100:.4f}%\n"
        f"Average swap fee percent: {pr_results['avg_swap_fee_rate'] * 100:.4f}%\n"
        f"Sharpe ratio: {pr_results['sharpe']:.4f}\n"
        f"Sharpe month ratio: {pr_results['sharpe_mon']:.4f}\n"
        f"Sortino ratio: {pr_results['sortino']:.4f}\n"
        f"Omega ratio: {pr_results['omega']:.4f}\n"
        f"Normalized Profit factor: {pr_results['nor_profit_fac']:.4f}\n"
        f"Normalized Pearson r: {pr_results['nor_coef_r']:.4f}\n"
        f"Van Tharp SQN: {pr_results['sqn']:.4f}\n"
        f"Sharpe SQN: {pr_results['sqn_by_sharpe']:.4f}\n"
        f"Sortino SQN: {pr_results['sqn_by_sortino']:.4f}\n"
        f"Normalized Edge ratio: {pr_results['nor_edge_ratio']:.4f}\n"
        f"Exit Efficiency: {pr_results['exit_efficiency']:.4f}\n"
        f"Max drawdown: {pr_results['max_drawdown'] * 100:.4f}%\n"
        f"Mean drawdown: {pr_results['mean_drawdown'] * 100:.4f}%\n"
        f"CAGR: {pr_results['cagr']:.4f}\n"
        f"Return/Max drawdown: {pr_results['return_by_maxdd']:.4f}\n"
        f"Return/Mean drawdown: {pr_results['return_by_meandd']:.4f}\n"
        f"Long profit percent: {pr_results['long_profit_percent'] * 100:.4f}%\n"
        f"Long loss percent: {pr_results['long_loss_percent'] * 100:.4f}%\n"
        f"Net worth: {pr_results['net_worth'] * 100:.8f}%\n"
    )
    if print_on==1:
        print(text)

    if result_txt_path is not None:
        with open(result_txt_path, 'w') as fo:
            fo.write(text)

    if result_png_path is not None:
        plt.ioff()
        fig, ax = plt.subplots()
        ax.axis("off")
        txt_obj = ax.text(0, 1, text, fontsize=12, fontfamily="monospace", va="top")
        renderer = fig.canvas.get_renderer()
        bbox = txt_obj.get_window_extent(renderer=renderer)

        dpi = 300
        width_in = bbox.width / dpi
        height_in = bbox.height / dpi
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
        ax.axis("off")
        ax.text(0, 1, text, fontsize=12, fontfamily="monospace", va="top")
        plt.savefig(result_png_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close()


def plot_eq(eq_series):
    plt.ioff()
    fig, ax = plt.subplots(figsize=(12, 6))
    date_format = mpl_dates.DateFormatter('%Y-%m-%d')
    Date_Render_range = eq_series.index.values
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    ax.set_ylabel('Balance')
    ax.plot(Date_Render_range, eq_series.values)

    return fig, ax


def plot_eq_datr(eq_series, agent_dict, DATR_LEN=22, COMPARE_BH=0):
    plt.ioff()

    # Get datr
    agent_dict_copy = deepcopy(agent_dict)
    agent_dict_copy['general']['USED_TFS'] = '1d'
    agent_dict_copy['general']['TF_OFFSET'] = 0
    agent_dict_copy['general']['BUFSIZE'] = int(DATR_LEN*3)
    df = read_df_sets(agent_dict_copy)[0][0]
    datr = AverageTrueRange(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        window=DATR_LEN
    ).average_true_range()
    datr_new_idx = get_leftmost_time_idx(eq_series.index, datr.index) - 1
    datr = datr.iloc[datr_new_idx]
    datr.index = eq_series.index
    datr = datr.dropna(inplace=False)

    # Plot
    fig, axs = plt.subplots(
        2, 1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [2, 1]}
    )

    # Format x-axis as date
    date_format = mpl_dates.DateFormatter('%Y-%m-%d')
    axs[0].xaxis.set_major_formatter(date_format)
    axs[1].xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    # Plot equity curve
    axs[0].set_ylabel('Balance')
    axs[0].plot(eq_series.index, eq_series.values, label='Equity Curve')

    # Plot datr series
    axs[1].set_ylabel('DATR')
    axs[1].plot(datr.index, datr.values, label='DATR', color='orange')
    plt.tight_layout()

    ######### Compare with B&H
    if COMPARE_BH==1:
        df_close = read_df_sets(agent_dict)[0][0]['Close']
        common_index = df_close.index.intersection(eq_series.index).sort_values()
        df_close = df_close.reindex(common_index).ffill()
        # df_close start end map to eq_series, by chatgpt
        if eq_series.iloc[-1]>eq_series.iloc[0] and df_close.iloc[-1]>df_close.iloc[0]:
            df_close = (df_close - df_close.iloc[0]) * (eq_series.iloc[-1] - eq_series.iloc[0]) / (df_close.iloc[-1] - df_close.iloc[0]) + eq_series.iloc[0]
        else:
            df_close = df_close * (eq_series.iloc[0]/df_close.iloc[0])
        axs[0].plot(df_close, label='B&H')
        axs[0].legend()

    return fig, axs


def normalize_orders_info(orders_info):
    # Only consider closed pos, and normalized the pnl
    orders_info_noc = orders_info.loc[~(orders_info['cancel']==1)]
    orders_info_noc_pair =orders_info_noc.groupby('position_cnt').filter(lambda x: len(x) >= 2)
    orders_info_noc_close_pos = orders_info_noc_pair.loc[orders_info_noc_pair['base']==0].copy()
    orders_info_noc_close_pos['margin'] = abs(orders_info_noc_pair.iloc[1::2]['exec_base'].values) * orders_info_noc_pair.iloc[::2]['exec_price'].values
    if len(orders_info_noc_close_pos)!=0:
        initial_margin = abs(orders_info_noc_close_pos['margin'].iloc[0])
    else:
        initial_margin = 0
    orders_info_noc_close_pos['pnl'] = orders_info_noc_close_pos['pnl']*initial_margin / orders_info_noc_close_pos['margin']
    orders_info_noc_close_pos['trade_fee'] = orders_info_noc_close_pos['trade_fee']*initial_margin / orders_info_noc_close_pos['margin']
    orders_info_noc_close_pos['spread_fee'] = orders_info_noc_close_pos['spread_fee']*initial_margin / orders_info_noc_close_pos['margin']
    orders_info_noc_close_pos['slippage_fee'] = orders_info_noc_close_pos['slippage_fee']*initial_margin / orders_info_noc_close_pos['margin']
    orders_info_noc_close_pos['swap_fee'] = orders_info_noc_close_pos['swap_fee']*initial_margin / orders_info_noc_close_pos['margin']
    orders_info_noc_close_pos['mfe'] = orders_info_noc_close_pos['mfe']*initial_margin / orders_info_noc_close_pos['margin']
    orders_info_noc_close_pos['mae'] = orders_info_noc_close_pos['mae']*initial_margin / orders_info_noc_close_pos['margin']
    nor_orders_info = deepcopy(orders_info)
    nor_orders_info.loc[orders_info_noc_close_pos.index, :] = orders_info_noc_close_pos
    return nor_orders_info


def load_config_xlsx(CONFIG_FILE):
    ### Get config content
    cfgs = pd.read_excel(CONFIG_FILE, sheet_name=None)

    ### Get general config
    general_cfgs = cfgs['general'].to_dict('records')

    ### Make agent_dicts_dict start from general config
    agent_dicts_dict = {general_cfg['AGENT']: dict(general=general_cfg) for general_cfg in general_cfgs}

    ### Get other config
    sheet_names = list(cfgs.keys())
    other_cfg_names = ['strategy', 'backtest', 'opt_params', 'walk_forward', 'live', 'live_intf']
    for cfg_name in other_cfg_names:
        # Get all names
        cfg_names = [x for x in sheet_names if x.startswith(cfg_name)]

        # Special handle for live cause confuse with live_intf, also initialized intf dict array for each agent_dict
        if cfg_name=='live' and 'live' in sheet_names:
            cfg_names = [cfg_name]
            agent_dicts_dict = {k: {**v, "live_intf": {}} for k, v in agent_dicts_dict.items()}

        # Loop cfg and add to agent_dicts_dict
        for cfg_name_full in cfg_names:
            cfgs_list = cfgs[cfg_name_full].to_dict('records')
            for cfg_dict in cfgs_list:
                if cfg_name != 'live_intf':
                    agent_dicts_dict[cfg_dict['AGENT']][cfg_name] = cfg_dict
                else:
                    agent_dicts_dict[cfg_dict['AGENT']]['live_intf'][cfg_dict['TRADE_INTERFACE']] = cfg_dict

    return agent_dicts_dict


def store_config_xlsx(agent_dicts_dict, CONFIG_FILE):
    # Remove
    delete_file(CONFIG_FILE)

    # Get all function names
    func_names = set([agent_dict['general']['FUNCTION'] for agent_dict in agent_dicts_dict.values()]) # use set to remove repeated one

    # Store config
    cfg_names = ['general', 'strategy', 'backtest', 'opt_params', 'walk_forward', 'live']
    with pd.ExcelWriter(CONFIG_FILE, mode='w') as writer:
        for cfg_name in cfg_names:
            # Store no diff cfg
            if cfg_name in ['general', 'backtest', 'walk_forward', 'live']:
                cfg = pd.DataFrame([agent_dict[cfg_name] for agent_dict in agent_dicts_dict.values() if cfg_name in agent_dict.keys()])
                if len(cfg)!=0:
                    cfg.to_excel(writer, index=False, header=True, sheet_name=cfg_name)

            # Store function diff cfg
            if cfg_name in ['strategy', 'opt_params']:
                for func_name in func_names:
                    agent_dicts_list = [agent_dict for agent_dict in agent_dicts_dict.values() if agent_dict['general']['FUNCTION']==func_name]
                    cfg_list = []
                    for agent_dict in agent_dicts_list:
                        if cfg_name in agent_dict.keys():
                            cfg_list.append(agent_dict[cfg_name])

                    cfg = pd.DataFrame(cfg_list)
                    if len(cfg)!=0:
                        cfg.to_excel(writer, index=False, header=True, sheet_name=cfg_name+'_'+func_name)


def load_symbols_info_xlsx(agent_dicts_dict, SYMBOLS_INFO_FILES):
    if isinstance(SYMBOLS_INFO_FILES, list):
        matching_files = SYMBOLS_INFO_FILES
    else:
        matching_files = glob.glob(SYMBOLS_INFO_FILES)

    symbols_info_list = []
    for file_name in matching_files:
        symbols_info_list += pd.read_excel(file_name, sheet_name="symbols_info").to_dict('records')

    # Add symbol info agent_dicts_dict
    symbols_info_dict = {symbol_info['SYMBOL']: symbol_info for symbol_info in symbols_info_list}
    for agent_name, agent_dict in agent_dicts_dict.items():
        symbols = agent_dict['general']['SYMBOLS'].split(",")
        symbols_info_part = [deepcopy(symbols_info_dict[symbol]) for symbol in symbols]
        agent_dicts_dict[agent_name]['symbols_info'] = symbols_info_part

    return agent_dicts_dict


def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def augment_str_decode(augment):
    pairs = re.findall(r'\[(.*?)\]', augment)
    result = []
    remove_top_per = 0
    for pair in pairs:
        if pair=='':
            continue
        elements = pair.split(',')
        if elements[0] == 'SHIFT_TIME':
            result.append({'method':elements[0], 'shift_tf': elements[1], 'repeats': int(elements[2])})
        elif elements[0] == 'ATR_NOISE':
            result.append({'method': elements[0], 'atr_noise': float(elements[1]), 'repeats': int(elements[2])})
        elif elements[0] == 'REMOVE_TOP':
            remove_top_per = float(elements[1])
        else:
            raise ValueError("Invalid augment method " + elements[0])

    return result, remove_top_per


def delete_file(filepath):
    if os.path.isfile(filepath):
        os.chmod(filepath, stat.S_IWRITE)
        os.remove(filepath)
    elif os.path.isdir(filepath):
        while 1:
            if not os.path.exists(filepath):
                break
            try:
                shutil.rmtree(filepath)
            except PermissionError as e:
                err_file_path = str(e).split("\'", 2)[1]
                if os.path.exists(err_file_path):
                    os.chmod(err_file_path, stat.S_IWRITE)