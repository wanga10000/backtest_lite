import numpy as np
import pandas as pd
from function.function_base import function_base
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, ADXIndicator


class all_grid(function_base):
    def __init__(self, agent_name, log_print):
        self.agent_name = agent_name
        self.log_print = log_print     # args: (str, to_tg=0)


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
        self.EMA_LEN = int(strategy_cfg['EMA_LEN'])
        self.ADX_LEN = int(strategy_cfg['ADX_LEN'])
        self.ADX_THD = float(strategy_cfg['ADX_THD'])
        self.GRID_CNTU = int(strategy_cfg['GRID_CNTU'])
        self.GRID_CNTD = int(strategy_cfg['GRID_CNTD'])
        self.GRID_PER = float(strategy_cfg['GRID_PER'])
        self.HEDGE_RAT = float(strategy_cfg['HEDGE_RAT'])
        self.CLOSE_HEDGE_W1S = int(strategy_cfg['CLOSE_HEDGE_W1S'])

        self.first = 1
        self.has_hedge = self.HEDGE_RAT>0
        self.grid_cnt = self.GRID_CNTU + self.GRID_CNTD
        self.hedging = 0


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
        self.dfh = df_set[1]
        self.dfm = df_set[0]


    def indicator_calc(self):
        if len(self.dfh) >= self.BUFSIZE[1]:  # A secure to make sure data satisfy BUFSIZE
            # ATR for risk
            self.dfh['ATR'] = AverageTrueRange(high=self.dfh['High'],
                                               low=self.dfh['Low'],
                                               close=self.dfh['Close'],
                                               window=40).average_true_range()

            # EMA
            self.dfh['EMA'] = EMAIndicator(close=self.dfh['Close'],
                                           window=self.EMA_LEN).ema_indicator()

            # ADX
            adx_indicator = ADXIndicator(
                high=self.dfh['High'],
                low=self.dfh['Low'],
                close=self.dfh['Close'],
                window=self.ADX_LEN
            )
            adx = adx_indicator.adx()
            plus_di = adx_indicator.adx_pos()
            minus_di = adx_indicator.adx_neg()
            self.dfh['ADX_DOWN'] = (adx > self.ADX_THD) & (minus_di > plus_di)
            self.dfh['ADX_UP'] = (adx > self.ADX_THD) & (minus_di < plus_di)


    def on_tick(self, cur_sidx, cur_date_int, bid, ask, kidx_sets, kidx_changed_flags, pending_orders, cur_positions, cur_balance):
        new_orders = []
        cancel_order_ids = []

        ### W1S trigger
        if self.CLOSE_HEDGE_W1S==1:
            if self.has_hedge==1 and (-1 in cur_positions.keys()):
                # Close reverse position
                new_orders += [dict(position_id = -1, order_id = 2)]
            self.CLOSE_HEDGE_W1S = 0

        ### When dfh changed, means dfm also change
        if kidx_changed_flags[cur_sidx][1]==1:
            idxh = kidx_sets[0][1]

            ### if price touch hedge, open hedge position
            if self.has_hedge:
                if -1 not in cur_positions.keys() and cur_positions and self.PAUSE==0:
                    if (self.SIDE==1 and self.dfh['ADX_DOWN'][idxh]) or (self.SIDE==2 and self.dfh['ADX_UP'][idxh]):
                        base_sum = sum([pos_info['base'] for pos_info in cur_positions.values()]) * self.HEDGE_RAT
                        base_side = 1 if base_sum>=0 else -1
                        base_sum = (abs(base_sum) // self.BASE_STEP) * self.BASE_STEP if self.BASE_STEP!=0 else abs(base_sum)
                        base_sum = base_side * base_sum
                        new_orders += [dict(position_id = -1, order_id = 2, base = -base_sum)]
                        self.hedging = 1
                elif -1 in cur_positions.keys():
                    if (self.SIDE==1 and not self.dfh['ADX_DOWN'][idxh]) or (self.SIDE==2 and not self.dfh['ADX_UP'][idxh]):
                        # Close reverse position
                        new_orders += [dict(position_id = -1, order_id = 2)]
                        self.hedging = 0

            ### Calculate grid
            grid_width = self.dfh['EMA'][idxh] * (self.GRID_PER/100)
            lower_bound = self.dfh['EMA'][idxh] - self.GRID_CNTD * grid_width
            self.grids = np.array([lower_bound + grid_width * gidx for gidx in range(self.grid_cnt + 1)])
            self.first = 0

            ### Grid logic
            # Remove previous limit order and limit profit order
            if pending_orders[0]:
                cancel_order_ids += [0, 1]

            # Get grids ask
            grids_ask = self.grids + (ask - bid)

            # SIDE=1 Grid open
            if self.SIDE==1:
                # Get current pos by the position id
                pos_open_levels = grids_ask[0:-1]
                pos_close_levels = self.grids[1:]

                # Add positions
                if self.PAUSE==0 and self.hedging==0:
                    if ask <= pos_close_levels[-1]: # Only open in range
                        cur_loc = np.searchsorted(pos_open_levels, ask, side="left") - 1

                        if cur_loc!=-1 and (cur_loc not in cur_positions.keys()):
                            base, broken = self.base_calc(idxh, cur_balance, ask)

                            if not broken:
                                profit_order = dict(position_id = cur_loc, order_type = 1, order_id = 1, price = pos_close_levels[cur_loc])
                                new_orders += [dict(position_id = cur_loc, order_type = 1, order_id = 0, price = pos_open_levels[cur_loc], base = base, oco_orders = [profit_order])]

                # Add new limit profit order for cur position
                for pos_id in cur_positions.keys():
                    if pos_id != -1: #-1 is hedge position
                        new_orders += [dict(position_id = pos_id, order_type = 1, order_id = 1, price = pos_close_levels[pos_id])]

            # SIDE=2 Grid open
            elif self.SIDE==2:
                # Get current pos by the position id
                pos_open_levels = self.grids[1:]
                pos_close_levels = grids_ask[0:-1]

                # Add positions
                if self.PAUSE==0 and self.hedging==0:
                    if bid >= pos_close_levels[0]: # Only open in range
                        cur_loc = np.searchsorted(pos_open_levels, bid, side="right")

                        if cur_loc!=len(pos_open_levels) and (cur_loc not in cur_positions.keys()):
                            base, broken = self.base_calc(idxh, cur_balance, ask)

                            if not broken:
                                # Add limit order
                                profit_order = dict(position_id = cur_loc, order_type = 1, order_id = 1, price = pos_close_levels[cur_loc])
                                new_orders += [dict(position_id = cur_loc, order_type = 1, order_id = 0, price = pos_open_levels[cur_loc], base = -base, oco_orders = [profit_order])]

                # Add new limit profit order for cur position
                for pos_id in cur_positions.keys():
                    if pos_id != -1: #-1 is hedge position
                        new_orders += [dict(position_id = pos_id, order_type = 1, order_id = 1, price = pos_close_levels[pos_id])]

                # Log print for kidx change
                self.log_print(f"INFO [{self.agent_name}]: {pd.to_datetime(cur_date_int).strftime('%Y/%m/%d %H:%M:%S')} Current Balance: {cur_balance:.2f}")

        ### dfm change
        elif kidx_changed_flags[cur_sidx][0]==1:
            idxh = kidx_sets[0][1]

            ### initial
            if self.first==1:
                ### Calculate grid
                grid_width = self.dfh['EMA'][idxh] * (self.GRID_PER/100)
                lower_bound = self.dfh['EMA'][idxh] - self.GRID_CNTD * grid_width
                self.grids = np.array([lower_bound + grid_width * gidx for gidx in range(self.grid_cnt + 1)])
                self.first = 0

            ### if price touch hedge, open hedge position (for CLOSE_HEDGE_W1S repoen)
            if self.has_hedge:
                if -1 not in cur_positions.keys() and cur_positions and self.PAUSE==0:
                    if (self.SIDE==1 and self.dfh['ADX_DOWN'][idxh]) or (self.SIDE==2 and self.dfh['ADX_UP'][idxh]):
                        base_sum = sum([pos_info['base'] for pos_info in cur_positions.values()]) * self.HEDGE_RAT
                        base_side = 1 if base_sum>=0 else -1
                        base_sum = (abs(base_sum) // self.BASE_STEP) * self.BASE_STEP if self.BASE_STEP!=0 else abs(base_sum)
                        base_sum = base_side * base_sum
                        new_orders += [dict(position_id = -1, order_id = 2, base = -base_sum)]
                        self.hedging = 1
                elif -1 in cur_positions.keys():
                    if (self.SIDE==1 and not self.dfh['ADX_DOWN'][idxh]) or (self.SIDE==2 and not self.dfh['ADX_UP'][idxh]):
                        # Close reverse position
                        new_orders += [dict(position_id = -1, order_id = 2)]
                        self.hedging = 0

            ### SIDE=1 Grid open
            grids_ask = self.grids + (ask - bid)

            # Get current limit orders
            limit_orders = [porder for porder in pending_orders[0] if (porder['order_id']==0)]

            if self.SIDE==1:
                # Get current pos by the position id
                pos_open_levels = grids_ask[0:-1]
                pos_close_levels = self.grids[1:]

                # Add positions
                if self.PAUSE==0 and self.hedging==0:
                    if ask <= pos_close_levels[-1]: # Only open in range
                        cur_loc = np.searchsorted(pos_open_levels, ask, side="left") - 1
                        limit_idx = [order['position_id'] for order in limit_orders] if limit_orders else []

                        if cur_loc!=-1 and (cur_loc not in cur_positions.keys()) and (cur_loc not in limit_idx):
                            base, broken = self.base_calc(idxh, cur_balance, ask)

                            if not broken:
                                profit_order = dict(position_id = cur_loc, order_type = 1, order_id = 1, price = pos_close_levels[cur_loc])
                                new_orders += [dict(position_id = cur_loc, order_type = 1, order_id = 0, price = pos_open_levels[cur_loc], base = base, oco_orders = [profit_order])]

            ### SIDE=2 Grid open
            elif self.SIDE==2:
                # Get current pos by the position id
                pos_open_levels = self.grids[1:]
                pos_close_levels = grids_ask[0:-1]

                # Add positions
                if self.PAUSE==0 and self.hedging==0:
                    if bid >= pos_close_levels[0]: # Only open in range
                        cur_loc = np.searchsorted(pos_open_levels, bid, side="right")
                        limit_idx = [order['position_id'] for order in limit_orders] if limit_orders else []

                        if cur_loc!=len(pos_open_levels) and (cur_loc not in cur_positions.keys()) and (cur_loc not in limit_idx):
                            base, broken = self.base_calc(idxh, cur_balance, ask)

                            if not broken:
                                profit_order = dict(position_id = cur_loc, order_type = 1, order_id = 1, price = pos_close_levels[cur_loc])
                                new_orders += [dict(position_id = cur_loc, order_type = 1, order_id = 0, price = pos_open_levels[cur_loc], base = -base, oco_orders = [profit_order])]

            # Log print for kidx change
            self.log_print(f"INFO [{self.agent_name}]: {pd.to_datetime(cur_date_int).strftime('%Y/%m/%d %H:%M:%S')} Current Balance: {cur_balance:.2f}")

        return new_orders, cancel_order_ids


    def base_calc(self, idxh, cur_balance, price):
        broken = False

        # Get quote by POS_MODE
        if self.POS_MODE==0:
            quote = self.LEVERAGE * self.BALANCE
        elif self.POS_MODE==1:
            quote = self.LEVERAGE * cur_balance
        else:
            quote = ((self.RISK_PER / 100) / (self.dfh['ATR'][idxh]/price)) * cur_balance

        quote = quote / (self.grid_cnt)
        base = quote / price
        base = (base // self.BASE_STEP) * self.BASE_STEP if self.BASE_STEP!=0 else base
        quote_act = base * price

        # Check broken
        if base<=self.MIN_BASE-(1e-7) or quote_act<=self.MIN_QUOTE:
            broken = True

        return base, broken