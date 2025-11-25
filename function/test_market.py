import numpy as np
import pandas as pd
from function.function_base import function_base
from ta.volatility import AverageTrueRange


class test_market(function_base):
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
                side = self.check_open()

                ### if condition met, open order
                if side!=0:
                    # Find base
                    base, broken = self.base_calc(idx, side, cur_balance, bid, ask)

                    if not broken:
                        order_dict = dict(base = base, tag='open')
                        new_orders.append(order_dict)

            # If having position now, check if closing position
            else:
                # check closing position
                close_met = self.check_close()

                ### if condition met, add closing order
                if close_met!=0:
                    close_order_dict = dict(tag='close')
                    new_orders.append(close_order_dict)

                    ### if closed, recheck open
                    side = self.check_open()

                    ### if condition met, open order
                    if side!=0:
                        # Find base
                        base, broken = self.base_calc(idx, side, cur_balance, bid, ask)

                        if not broken:
                            order_dict = dict(base = base, tag='open')
                            new_orders.append(order_dict)

            # Log print for kidx change
            self.log_print(f"INFO [{self.agent_name}]: {pd.to_datetime(cur_date_int).strftime('%Y/%m/%d %H:%M:%S')} Current Balance: {cur_balance:.2f}")

        return new_orders, cancel_order_ids


    def check_open(self):
        # check opening side
        if self.PAUSE==1:
            return 0
        else:
            side = self.side_this_time
            self.side_this_time = -self.side_this_time

            return side


    def check_close(self):
        return 1


    def base_calc(self, idx, side, cur_balance, bid, ask):
        broken = False

        # Get open_price
        open_price = ask if side==1 else bid
        risk_price = open_price - side * self.df['ATR'][idx]

        # Get quote by POS_MODE
        if self.POS_MODE==0:
            quote = self.LEVERAGE * self.BALANCE
        elif self.POS_MODE==1:
            quote = self.LEVERAGE * cur_balance
        else:
            quote = ((self.RISK_PER / 100) / abs(1 - (risk_price/open_price))) * cur_balance

        base = quote / open_price
        base = (base // self.BASE_STEP) * self.BASE_STEP if self.BASE_STEP!=0 else base
        quote_act = base * open_price
        base = base * side

        # Check broken
        if abs(base)<=self.MIN_BASE-(1e-7) or quote_act<=self.MIN_QUOTE:
            broken = True

        return base, broken