import numpy as np
import pandas as pd
from function.function_base import function_base
from ta.volatility import AverageTrueRange


class test_limit(function_base):
    def __init__(self, agent_name, log_print):
        self.agent_name = agent_name
        self.log_print = log_print     # args: (str, to_tg=0)
        self.side_this_time = 1


    def load_general_cfg(self, general_cfg):
        self.BALANCE = float(general_cfg['BALANCE'])
        self.USED_TFS = str(general_cfg['USED_TFS']).split(",")
        self.BUFSIZE = list(map(int, str(general_cfg['BUFSIZE']).split(",")))
        self.PAUSE = int(general_cfg['PAUSE'])

        
    def load_strategy_cfg(self, strategy_cfg):
        self.main_tf_idx = self.USED_TFS.index(str(strategy_cfg['MAIN_TF']))
        self.POS_MODE = int(strategy_cfg['POS_MODE'])
        self.LEVERAGE = float(strategy_cfg['LEVERAGE'])
        self.RISK_PER = float(strategy_cfg['RISK_PER'])
        self.SIDE = int(strategy_cfg['SIDE'])
        self.ATR_LIMIT = float(strategy_cfg['ATR_LIMIT'])
        self.SP_VALUE = float(strategy_cfg['SP_VALUE'])


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
        self.df = df_set[self.main_tf_idx]


    def indicator_calc(self):
        if len(self.df) >= self.BUFSIZE[self.main_tf_idx]:  # A secure to make sure data satisfy BUFSIZE
            # Default Risk
            self.df['ATR'] = AverageTrueRange(high=self.df['High'],
                                              low=self.df['Low'],
                                              close=self.df['Close'],
                                              window=25).average_true_range()


    def on_tick(self, cur_sidx, cur_date_int, bid, ask, kidx_sets, kidx_changed_flags, pending_orders, cur_positions, cur_balance):
        new_orders = []
        cancel_order_ids = []

        # Check only when main df date change
        if kidx_changed_flags[0][self.main_tf_idx]==1:
            idx = kidx_sets[0][self.main_tf_idx]

            # If no position now, check if opening position
            if not cur_positions:
                # If previous limit order is not opening, close it
                cancel_order_ids = [order['order_id'] for order in pending_orders[0]]

                # check opening position
                side, order_type, price = self.check_open(idx, bid, ask)

                ### if condition met, open order
                if side!=0:
                    # Find S&P
                    stoploss, profit = self.cal_s_and_p(idx, side, order_type, price, bid, ask)

                    # Find base
                    base, broken = self.base_calc(side, cur_balance, order_type, price, bid, ask, stoploss)

                    if not broken:
                        # Create maker orders, oco_order_id 1 means stoploss order, 2 means profit order
                        stoploss_order_dict = dict(order_id = 1, order_type = 2, price = stoploss, oco_cancel_ids = [2], tag='stoploss')
                        profit_order_dict = dict(order_id = 2, order_type = 1, price = profit, oco_cancel_ids = [1], tag='takeprofit')

                        if order_type==0:
                            # If open market, add S&P orders on new_orders.
                            order_dict = dict(base = base, tag='open_market')
                            new_orders.append(order_dict)
                            new_orders.append(stoploss_order_dict)
                            new_orders.append(profit_order_dict)
                        else:
                            # If open limit, add S&P orders on its oco_orders
                            order_dict = dict(order_type = order_type, base = base, price = price, oco_orders = [stoploss_order_dict, profit_order_dict], tag='open_limit')
                            new_orders.append(order_dict)

            # Log print for kidx change
            self.log_print(f"INFO [{self.agent_name}]: {pd.to_datetime(cur_date_int).strftime('%Y/%m/%d %H:%M:%S')} Current Balance: {cur_balance:.2f}")

        return new_orders, cancel_order_ids


    def check_open(self, idx, bid, ask):
        side = self.side_this_time
        self.side_this_time = -self.side_this_time

        if self.PAUSE==1:
            return 0, 0, 0
        else:
            if side==1:
                if self.ATR_LIMIT<=0:
                    order_type = 0
                    price = 0
                else:
                    order_type = 1
                    price = ask - self.ATR_LIMIT * self.df['ATR'][idx]

                return 1, order_type, price
            else:
                if self.ATR_LIMIT<=0:
                    order_type = 0
                    price = 0
                else:
                    order_type = 1
                    price = bid + self.ATR_LIMIT * self.df['ATR'][idx]

                return -1, order_type, price


    def cal_s_and_p(self, idx, side, order_type, price, bid, ask):
        if order_type == 0:
            baseline_price = ask if side==-1 else bid
        else:
            baseline_price = price

        stoploss = baseline_price - side * self.SP_VALUE * self.df['ATR'][idx]
        profit = baseline_price + side * self.SP_VALUE * self.df['ATR'][idx]
        return stoploss, profit


    def base_calc(self, side, cur_balance, order_type, price, bid, ask, stoploss):
        broken = False

        # Get open_price
        if order_type == 0:
            open_price = ask if side==1 else bid
        else:
            open_price = price

        # Get quote by POS_MODE
        if self.POS_MODE==0:
            quote = self.LEVERAGE * self.BALANCE
        elif self.POS_MODE==1:
            quote = self.LEVERAGE * cur_balance
        else:
            quote = ((self.RISK_PER / 100) / abs(1 - (stoploss/open_price))) * cur_balance

        base = quote / open_price
        base = (base // self.BASE_STEP) * self.BASE_STEP if self.BASE_STEP!=0 else base
        quote_act = base * open_price
        base = base * side

        # Check broken
        if abs(base)<=self.MIN_BASE-(1e-7) or quote_act<=self.MIN_QUOTE:
            broken = True

        return base, broken