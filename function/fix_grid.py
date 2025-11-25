import numpy as np
import pandas as pd
from function.function_base import function_base
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, ADXIndicator


class fix_grid(function_base):
    def __init__(self, agent_name, log_print):
        self.agent_name = agent_name
        self.log_print = log_print     # args: (str, to_tg=0)
        self.touched_bound = 0


    def load_general_cfg(self, general_cfg):
        self.BALANCE = float(general_cfg['BALANCE'])
        self.USED_TFS = str(general_cfg['USED_TFS']).split(",")
        self.BUFSIZE = list(map(int, str(general_cfg['BUFSIZE']).split(",")))
        self.PAUSE = int(general_cfg['PAUSE'])


    def load_strategy_cfg(self, strategy_cfg):
        self.POS_MODE = int(strategy_cfg['POS_MODE'])
        self.LEVERAGE = float(strategy_cfg['LEVERAGE'])
        self.RISK_PER = float(strategy_cfg['RISK_PER'])

        self.SIDE = int(strategy_cfg['SIDE'])   # Only 1 or 2
        self.GRID_CNT = int(strategy_cfg['GRID_CNT'])
        self.GRID_UPPER = float(strategy_cfg['GRID_UPPER'])
        self.GRID_LOWER = float(strategy_cfg['GRID_LOWER'])
        self.TOUCH_BOUND_STOP = int(strategy_cfg['TOUCH_BOUND_STOP'])

        grid_width = (self.GRID_UPPER - self.GRID_LOWER) / self.GRID_CNT
        self.grids = np.array([self.GRID_LOWER + grid_width * gidx for gidx in range(self.GRID_CNT + 1)])


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
        self.dfm = df_set[0]


    def indicator_calc(self):
        pass


    def on_tick(self, cur_sidx, cur_date_int, bid, ask, kidx_sets, kidx_changed_flags, pending_orders, cur_positions, cur_balance):
        new_orders = []
        cancel_order_ids = []

        ### If touch bound, this agent already died, do not open new positions.
        if self.touched_bound==0:
            ### dfm change
            if kidx_changed_flags[cur_sidx][0]==1:
                ### SIDE=1 Grid open
                grids_ask = self.grids + (ask - bid)

                # Get current limit orders
                limit_orders = [porder for porder in pending_orders[0] if (porder['order_id']==0)]

                if self.SIDE==1:
                    # Get current pos by the position id
                    pos_open_levels = grids_ask[0:-1]
                    pos_close_levels = self.grids[1:]

                    # Check touch bound
                    if self.TOUCH_BOUND_STOP==1:
                        if ask <= pos_open_levels[0] or bid >= pos_close_levels[-1]:
                            self.touched_bound = 1
                            # Close all positions
                            new_orders = [{'symbol_idx': pos_info['symbol_idx'], 'position_id': pos_idx} for pos_idx, pos_info in cur_positions.items()]
                            cancel_order_ids = [porder['order_id'] for porder in pending_orders[0]]
                            return new_orders, cancel_order_ids

                    # Add positions
                    if self.PAUSE==0:
                        if ask <= pos_close_levels[-1]: # Only open in range
                            cur_loc = np.searchsorted(pos_open_levels, ask, side="left") - 1

                            # Handle market orders
                            cur_market_idxs = list(cur_positions.keys())
                            add_market_idxs = list(range(cur_loc+1, self.GRID_CNT))
                            add_market_idxs = sorted(list((set(add_market_idxs) - set(cur_market_idxs))))

                            # Handle limit order
                            cur_limit_idx = [order['position_id'] for order in limit_orders] if limit_orders else []
                            has_limit = cur_loc!=-1 and (cur_loc not in cur_positions.keys()) and (cur_loc not in cur_limit_idx)

                            # Add market & limit orders
                            if add_market_idxs or has_limit:
                                base, broken = self.base_calc(cur_balance, ask)
                                if not broken:
                                    # Add market orders
                                    if add_market_idxs:
                                        for aidx in add_market_idxs:
                                            new_orders += [dict(position_id = aidx, order_id = 0, price = pos_open_levels[aidx], base = base)]
                                            new_orders += [dict(position_id = aidx, order_type = 1, order_id = 1, price = pos_close_levels[aidx])]

                                    # Add limit order
                                    if has_limit:
                                        profit_order = dict(position_id = cur_loc, order_type = 1, order_id = 1, price = pos_close_levels[cur_loc])
                                        new_orders += [dict(position_id = cur_loc, order_type = 1, order_id = 0, price = pos_open_levels[cur_loc], base = base, oco_orders = [profit_order])]

                ### SIDE=2 Grid open
                elif self.SIDE==2:
                    # Get current pos by the position id
                    pos_open_levels = self.grids[1:]
                    pos_close_levels = grids_ask[0:-1]

                    # Check touch bound
                    if self.TOUCH_BOUND_STOP==1:
                        if bid >= pos_open_levels[-1] or ask <= pos_open_levels[0]:
                            self.touched_bound = 1
                            # Close all positions
                            new_orders = [{'symbol_idx': pos_info['symbol_idx'], 'position_id': pos_idx} for pos_idx, pos_info in cur_positions.items()]
                            cancel_order_ids = [porder['order_id'] for porder in pending_orders[0]]
                            return new_orders, cancel_order_ids

                    # Add positions
                    if self.PAUSE==0:
                        if bid >= pos_close_levels[0]: # Only open in range
                            cur_loc = np.searchsorted(pos_open_levels, bid, side="right")

                            # Handle market orders
                            cur_market_idxs = list(cur_positions.keys())
                            add_market_idxs = list(range(0, cur_loc))
                            add_market_idxs = sorted(list((set(add_market_idxs) - set(cur_market_idxs))))

                            # Handle limit order
                            cur_limit_idx = [order['position_id'] for order in limit_orders] if limit_orders else []
                            has_limit = cur_loc!=len(pos_open_levels) and (cur_loc not in cur_positions.keys()) and (cur_loc not in cur_limit_idx)

                            # Add market & limit orders
                            if add_market_idxs or has_limit:
                                base, broken = self.base_calc(cur_balance, ask)
                                if not broken:
                                    # Add market orders
                                    if add_market_idxs:
                                        for aidx in add_market_idxs:
                                            new_orders += [dict(position_id = aidx, order_id = 0, price = pos_open_levels[aidx], base = -base)]
                                            new_orders += [dict(position_id = aidx, order_type = 1, order_id = 1, price = pos_close_levels[aidx])]

                                    # Add limit order
                                    if has_limit:
                                        profit_order = dict(position_id = cur_loc, order_type = 1, order_id = 1, price = pos_close_levels[cur_loc])
                                        new_orders += [dict(position_id = cur_loc, order_type = 1, order_id = 0, price = pos_open_levels[cur_loc], base = -base, oco_orders = [profit_order])]

                # Log print for kidx change
                self.log_print(f"INFO [{self.agent_name}]: {pd.to_datetime(cur_date_int).strftime('%Y/%m/%d %H:%M:%S')} Current Balance: {cur_balance:.2f}")

        return new_orders, cancel_order_ids


    def base_calc(self, cur_balance, price):
        broken = False

        # Get quote by POS_MODE
        if self.POS_MODE==0:
            quote = self.LEVERAGE * self.BALANCE
        elif self.POS_MODE==1:
            quote = self.LEVERAGE * cur_balance

        quote = quote / (self.GRID_CNT)
        base = quote / price
        base = (base // self.BASE_STEP) * self.BASE_STEP if self.BASE_STEP!=0 else base
        quote_act = base * price

        # Check broken
        if base<=self.MIN_BASE-(1e-7) or quote_act<=self.MIN_QUOTE:
            broken = True

        return base, broken