import numpy as np
import pandas as pd
from function.function_base import function_base
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator
from ta.trend import CCIIndicator
from datetime import datetime


class fox_orb(function_base):
    def __init__(self, agent_name, log_print):
        self.agent_name = agent_name
        self.log_print = log_print     # args: (str, to_tg=0)
        self.state = 0
        self.low_bid = 0
        self.high_ask = 0


    def load_general_cfg(self, general_cfg):
        self.BALANCE = float(general_cfg['BALANCE'])
        self.USED_TFS = str(general_cfg['USED_TFS']).split(",")
        self.BUFSIZE = list(map(int, str(general_cfg['BUFSIZE']).split(",")))
        self.PAUSE = int(general_cfg['PAUSE'])


    def load_strategy_cfg(self, strategy_cfg):
        self.POS_MODE = int(strategy_cfg['POS_MODE'])
        self.LEVERAGE = float(strategy_cfg['LEVERAGE'])
        self.RISK_PER = float(strategy_cfg['RISK_PER'])

        self.SIDE = int(strategy_cfg['SIDE'])
        self.OR_START_HOUR = int(strategy_cfg['OR_START_HOUR'])
        self.OR_START_MIN = int(strategy_cfg['OR_START_MIN'])
        self.OR_END_HOUR = int(strategy_cfg['OR_END_HOUR'])
        self.OR_END_MIN = int(strategy_cfg['OR_END_MIN'])
        self.CLOSE_HOUR = int(strategy_cfg['CLOSE_HOUR'])
        self.CLOSE_MIN = int(strategy_cfg['CLOSE_MIN'])
        self.WED_OFF = int(strategy_cfg['WED_OFF'])
        self.MEA_DE = float(strategy_cfg['MEA_DE'])
        self.MEA_SL = float(strategy_cfg['MEA_SL'])
        self.MEA_TP = float(strategy_cfg['MEA_TP'])
        self.MEA_BS = float(strategy_cfg['MEA_BS'])
        self.MEA_LK = float(strategy_cfg['MEA_LK'])
        self.MEA_GA = float(strategy_cfg['MEA_GA'])

        self.has_long = self.SIDE==0 or self.SIDE==1
        self.has_short = self.SIDE==0 or self.SIDE==2


    def load_symbols_info(self, symbols_info):
        self.symbols = [sym_info['SYMBOL'] for sym_info in symbols_info]
        self.MIN_QUOTE = symbols_info[0]['MIN_QUOTE']
        self.MIN_BASE = symbols_info[0]['MIN_BASE']
        self.BASE_STEP = symbols_info[0]['BASE_STEP']


    def load_df_sets(self, df_sets):
        self.df_sets = df_sets
        self.declare_df()


    def declare_df(self):
        # Main symbol df handle
        df_set = self.df_sets[0]

        # Main tf handle
        self.df = df_set[1]


    def indicator_calc(self):
        pass


    def on_tick(self, cur_sidx, cur_date_int, bid, ask, kidx_sets, kidx_changed_flags, pending_orders, cur_positions, cur_balance):
        new_orders = []
        cancel_order_ids = []

        # Check only when min df date change
        if kidx_changed_flags[0][0]==1:
            if self.state==2:
                # post market (de) if current price touch breakout
                if ask>=self.high_ask and self.has_long:
                    long_de_price, long_stoploss, long_profit, self.long_start_bs, self.long_lk_price, self.long_start_ga = self.cal_mea_stuff(1, ask)
                    base, broken = self.base_calc(cur_balance, long_de_price, long_stoploss)
                    if not broken:
                        # SL, TP logic for exit_orders
                        if self.MEA_SL>0 and self.MEA_TP>0: # With SL+TP
                            stoploss_order_dict = dict(order_type = 2, price = long_stoploss, order_id = 1, oco_cancel_ids = [2])
                            profit_order_dict = dict(order_type = 1, price = long_profit, order_id = 2, oco_cancel_ids = [1])
                            exit_orders = [stoploss_order_dict, profit_order_dict]

                        elif self.MEA_SL>0 and self.MEA_TP<=0: # With SL
                            stoploss_order_dict = dict(order_type = 2, price = long_stoploss, order_id = 1)
                            exit_orders = [stoploss_order_dict]

                        elif self.MEA_SL<=0 and self.MEA_TP>0: # With TP
                            profit_order_dict = dict(order_type = 1, price = long_profit, order_id = 2)
                            exit_orders = [profit_order_dict]
                        else:
                            exit_orders = []

                        # DE logic for entry_order
                        if self.MEA_DE==0:
                            # If open market, add Stoploss orders on pending_orders
                            new_orders += [dict(base = base)]
                            new_orders += exit_orders
                        else:
                            # If open limit, add S&P orders on its oco_orders
                            new_orders += [dict(order_type = 1, base = base, price = long_de_price, oco_orders = exit_orders)]
                    self.state = 3

                elif bid<=self.low_bid and self.has_short:
                    short_de_price, short_stoploss, short_profit, self.short_start_bs, self.short_lk_price, self.short_start_ga = self.cal_mea_stuff(-1, bid)
                    base, broken = self.base_calc(cur_balance, short_de_price, short_stoploss)
                    if not broken:
                        # SL, TP logic for exit_orders
                        if self.MEA_SL>0 and self.MEA_TP>0: # With SL+TP
                            stoploss_order_dict = dict(order_type = 2, price = short_stoploss, order_id = 1, oco_cancel_ids = [2])
                            profit_order_dict = dict(order_type = 1, price = short_profit, order_id = 2, oco_cancel_ids = [1])
                            exit_orders = [stoploss_order_dict, profit_order_dict]

                        elif self.MEA_SL>0 and self.MEA_TP<=0: # With SL
                            stoploss_order_dict = dict(order_type = 2, price = short_stoploss, order_id = 1)
                            exit_orders = [stoploss_order_dict]

                        elif self.MEA_SL<=0 and self.MEA_TP>0: # With TP
                            profit_order_dict = dict(order_type = 1, price = short_profit, order_id = 2)
                            exit_orders = [profit_order_dict]
                        else:
                            exit_orders = []

                       # DE logic for entry_order
                        if self.MEA_DE==0:
                            # If open market, add Stoploss orders on pending_orders
                            new_orders += [dict(base = -base)]
                            new_orders += exit_orders
                        else:
                            # If open limit, add S&P orders on its oco_orders
                            new_orders += [dict(order_type = 1, base = -base, price = short_de_price, oco_orders = exit_orders)]
                    self.state = 3

            # Good add detect
            if self.MEA_GA>0:
                if self.state==3 and cur_positions and not self.gaing:
                    # Check time (before close_hour-2)
                    cur_date = pd.to_datetime(cur_date_int)
                    if cur_date.hour <= self.CLOSE_HOUR-3:
                        base = cur_positions[0]['base']
                        open_side = 1 if (base>0) else -1
                        if (open_side==1 and bid > self.long_start_ga) or (open_side==-1 and ask < self.short_start_ga):
                            # Main logic of ga
                            self.gaing = True
                            new_orders += [dict(position_id = 1, base = base)] # This base already have side
                            stoploss_order = [porder for porder in pending_orders[0] if porder['order_id']==1]
                            profit_order = [porder for porder in pending_orders[0] if porder['order_id']==2]

                            for sorder in stoploss_order:
                                new_orders += [dict(position_id = 1, order_type = 2, price = sorder['price'], order_id = 1, oco_cancel_ids = [2])]

                            for porder in profit_order:
                                new_orders += [dict(position_id = 1, order_type = 1, price = porder['price'], order_id = 2, oco_cancel_ids = [1])]

            # Breakeven stop detect
            if self.MEA_BS>0:
                if self.state==3 and cur_positions and not self.bsing and not self.gaing:
                    open_side = 1 if (cur_positions[0]['base']>0) else -1
                    if (open_side==1 and bid > self.long_start_bs) or (open_side==-1 and ask < self.short_start_bs):
                        # Main logic of lk
                        lk_price = self.long_lk_price if open_side==1 else self.short_lk_price
                        self.bsing = True
                        if self.MEA_SL>0 and self.MEA_TP>0: # With SL+TP, modify SL
                            cancel_order_ids = [1]
                            new_orders += [dict(order_type = 2, price = lk_price, order_id = 1, oco_cancel_ids = [2])]

                        elif self.MEA_SL>0 and self.MEA_TP<=0: # With SL, modify SL
                            cancel_order_ids = [1]
                            new_orders += [dict(order_type = 2, price = lk_price, order_id = 1)]

                        elif self.MEA_SL<=0 and self.MEA_TP>0: # With TP, modify SL+TP
                            cancel_order_ids = [1, 2]
                            org_profit = [porder for porder in pending_orders[0] if porder['order_id']==2][0]['price']
                            new_orders += [dict(order_type = 2, price = lk_price, order_id = 1, oco_cancel_ids = [2])]
                            new_orders += [dict(order_type = 1, price = org_profit, order_id = 2, oco_cancel_ids = [1])]
                        else:  # With nothing, add SL
                            new_orders += [dict(order_type = 2, price = lk_price, order_id = 1)]

            # Log print for kidx change
            self.log_print(f"INFO [{self.agent_name}]: {pd.to_datetime(cur_date_int).strftime('%Y/%m/%d %H:%M:%S')} Current Balance: {cur_balance:.2f}")

        # Check only when main df date change
        if kidx_changed_flags[0][1]==1:
            cur_date = pd.to_datetime(cur_date_int)

            # ORB state Logic:
            # 0. idle
            # 1. At or_start_time, start collecting low high,
            # 2. At or_end_time, open 2 trigger breakout order
            # 3. If position opened, close position at close_time
            # If not, delete orders at close_time

            # 0. idle
            if self.state==0:
                # Initial dates
                self.or_start_date = cur_date.replace(hour=self.OR_START_HOUR, minute=self.OR_START_MIN, second=0, microsecond=0, nanosecond=0)
                self.or_end_date = cur_date.replace(hour=self.OR_END_HOUR, minute=self.OR_END_MIN, second=0, microsecond=0, nanosecond=0)
                self.close_date = cur_date.replace(hour=self.CLOSE_HOUR, minute=self.CLOSE_MIN, second=0, microsecond=0, nanosecond=0)

                if cur_date>=self.or_start_date and (not self.WED_OFF or cur_date.weekday() != 2):
                    self.state = 1
                    self.high_ask = ask
                    self.low_bid = bid
                    self.bsing = False
                    self.gaing = False

            # 1. collecting low high
            elif self.state==1:
                if cur_date>=self.or_end_date:
                    self.state = 2

                    # open market (de) if current price already over boundary
                    if ask>=self.high_ask and self.has_long:
                        long_de_price, long_stoploss, long_profit, self.long_start_bs, self.long_lk_price, self.long_start_ga = self.cal_mea_stuff(1, ask)
                        base, broken = self.base_calc(cur_balance, long_de_price, long_stoploss)
                        if not broken:
                            # SL, TP logic for exit_orders
                            if self.MEA_SL>0 and self.MEA_TP>0: # With SL+TP
                                stoploss_order_dict = dict(order_type = 2, price = long_stoploss, order_id = 1, oco_cancel_ids = [2])
                                profit_order_dict = dict(order_type = 1, price = long_profit, order_id = 2, oco_cancel_ids = [1])
                                exit_orders = [stoploss_order_dict, profit_order_dict]

                            elif self.MEA_SL>0 and self.MEA_TP<=0: # With SL
                                stoploss_order_dict = dict(order_type = 2, price = long_stoploss, order_id = 1)
                                exit_orders = [stoploss_order_dict]

                            elif self.MEA_SL<=0 and self.MEA_TP>0: # With TP
                                profit_order_dict = dict(order_type = 1, price = long_profit, order_id = 2)
                                exit_orders = [profit_order_dict]
                            else:
                                exit_orders = []

                            # DE logic for entry_order
                            if self.MEA_DE==0:
                                # If open market, add Stoploss orders on pending_orders
                                new_orders += [dict(base = base)]
                                new_orders += exit_orders
                            else:
                                # If open limit, add S&P orders on its oco_orders
                                new_orders += [dict(order_type = 1, base = base, price = long_de_price, oco_orders = exit_orders)]
                        self.state = 3

                    elif bid<=self.low_bid and self.has_short:
                        short_de_price, short_stoploss, short_profit, self.short_start_bs, self.short_lk_price, self.short_start_ga = self.cal_mea_stuff(-1, bid)
                        base, broken = self.base_calc(cur_balance, short_de_price, short_stoploss)
                        if not broken:
                            # SL, TP logic for exit_orders
                            if self.MEA_SL>0 and self.MEA_TP>0: # With SL+TP
                                stoploss_order_dict = dict(order_type = 2, price = short_stoploss, order_id = 1, oco_cancel_ids = [2])
                                profit_order_dict = dict(order_type = 1, price = short_profit, order_id = 2, oco_cancel_ids = [1])
                                exit_orders = [stoploss_order_dict, profit_order_dict]

                            elif self.MEA_SL>0 and self.MEA_TP<=0: # With SL
                                stoploss_order_dict = dict(order_type = 2, price = short_stoploss, order_id = 1)
                                exit_orders = [stoploss_order_dict]

                            elif self.MEA_SL<=0 and self.MEA_TP>0: # With TP
                                profit_order_dict = dict(order_type = 1, price = short_profit, order_id = 2)
                                exit_orders = [profit_order_dict]
                            else:
                                exit_orders = []

                           # DE logic for entry_order
                            if self.MEA_DE==0:
                                # If open market, add Stoploss orders on pending_orders
                                new_orders += [dict(base = -base)]
                                new_orders += exit_orders
                            else:
                                # If open limit, add S&P orders on its oco_orders
                                new_orders += [dict(order_type = 1, base = -base, price = short_de_price, oco_orders = exit_orders)]
                        self.state = 3
                else:
                    self.high_ask = ask if ask>self.high_ask else self.high_ask
                    self.low_bid = bid if bid<self.low_bid else self.low_bid

            # 2. open trigger and wait open position or delete order
            elif self.state==2:
                if cur_positions:
                    self.state = 3
                elif cur_date>=self.close_date:
                    self.state = 0
                    cancel_order_ids = [order['order_id'] for order in pending_orders[0]] # delete pending order

            # 3. wait for close position
            elif self.state==3:
                if cur_date>=self.close_date:
                    self.state = 0
                    new_orders = [dict(position_id=pid) for pid in cur_positions.keys()]
                    cancel_order_ids = [order['order_id'] for order in pending_orders[0]] # delete pending order

            # Log print for kidx change
            self.log_print(f"INFO [{self.agent_name}]: {pd.to_datetime(cur_date_int).strftime('%Y/%m/%d %H:%M:%S')} Current Balance: {cur_balance:.2f}")

        return new_orders, cancel_order_ids


    def cal_mea_stuff(self, side, break_price):
        de_price = break_price - side * self.MEA_DE
        stoploss = de_price - side * self.MEA_SL
        profit = de_price + side * self.MEA_TP
        start_bs = de_price + side * self.MEA_BS
        lk_price = de_price + side * self.MEA_LK
        start_ga = de_price + side * self.MEA_GA
        return de_price, stoploss, profit, start_bs, lk_price, start_ga


    def base_calc(self, cur_balance, price, stoploss):
        broken = False

        # Get quote by POS_MODE
        if self.POS_MODE==0:
            quote = self.LEVERAGE * self.BALANCE
        elif self.POS_MODE==1:
            quote = self.LEVERAGE * cur_balance
        else:
            risk_gap = abs(price - stoploss)
            quote = ((self.RISK_PER / 100) / (risk_gap/price)) * cur_balance

        base = quote / price
        base = (base // self.BASE_STEP) * self.BASE_STEP if self.BASE_STEP!=0 else base
        quote_act = base * price

        # Check broken
        if base<=self.MIN_BASE-(1e-7) or quote_act<=self.MIN_QUOTE:
            broken = True

        return base, broken