import numpy as np
import pandas as pd
from function.function_base import function_base
from ta.volatility import AverageTrueRange, BollingerBands
from function.func_lib import multi_symbols_synchronizer
import matplotlib.pyplot as plt


class paxg_gold(function_base):
    def __init__(self, agent_name, log_print):
        self.agent_name = agent_name
        self.log_print = log_print     # args: (str, to_tg=0)


    def load_general_cfg(self, general_cfg):
        self.BALANCE = float(general_cfg['BALANCE'])
        self.USED_TFS = str(general_cfg['USED_TFS']).split(",")
        self.BUFSIZE = list(map(int, str(general_cfg['BUFSIZE']).split(",")))
        self.PAUSE = int(general_cfg['PAUSE'])


    def load_strategy_cfg(self, strategy_cfg):
        self.LEVERAGE = float(strategy_cfg['LEVERAGE'])

        self.SIDE = int(strategy_cfg['SIDE'])
        self.BB_LEN = int(strategy_cfg['BB_LEN'])
        self.BB_ENTRY = float(strategy_cfg['BB_ENTRY'])
        self.HOUR_OUT = int(strategy_cfg['HOUR_OUT'])


    def load_symbols_info(self, symbols_info):
        self.symbols = [sym_info['SYMBOL'] for sym_info in symbols_info]
        self.MIN_QUOTES = [sym_info['MIN_QUOTE'] for sym_info in symbols_info]
        self.MIN_BASES = [sym_info['MIN_BASE'] for sym_info in symbols_info]
        self.BASE_STEPS = [sym_info['BASE_STEP'] for sym_info in symbols_info]

        # multi symbols synchornizer initialize
        self.ms_sync = multi_symbols_synchronizer(len(symbols_info))


    def load_df_sets(self, df_sets):
        self.df_sets = df_sets
        self.declare_df()


    def declare_df(self):
        # Main tf handle
        self.dfh0 = self.df_sets[0][-1]
        self.dfh1 = self.df_sets[1][-1]


    def indicator_calc(self):
        ### Spread calc
        common_index = self.dfh0.index.union(self.dfh1.index).sort_values()
        dfh0_align = self.dfh0.reindex(common_index).ffill()
        dfh1_align = self.dfh1.reindex(common_index).ffill()
        sprh_close = dfh0_align['Close'] - dfh1_align['Close']

        if len(sprh_close) >= self.BUFSIZE[-1]:  # A secure to make sure data satisfy BUFSIZE
            ### Indicator calculation
            # Trigger
            bol_tmp = BollingerBands(close=sprh_close,
                                     window=self.BB_LEN,
                                     window_dev = self.BB_ENTRY)
            self.spr_entry_h = bol_tmp.bollinger_hband()
            self.spr_entry_l = bol_tmp.bollinger_lband()
            self.spr_middle = bol_tmp.bollinger_mavg()


    def on_tick(self, cur_sidx, cur_date_int, bid, ask, kidx_sets, kidx_changed_flags, pending_orders, cur_positions, cur_balance):
        new_orders = []
        cancel_order_ids = []

        # Store the latest bid, ask
        if cur_sidx==0:
            self.bid0 = bid
            self.ask0 = ask
        else:
            self.bid1 = bid
            self.ask1 = ask

        # Check only when main df date change
        if kidx_changed_flags[cur_sidx][0]==1:
            if self.ms_sync.act(cur_sidx, cur_date_int):
                idx_date = pd.to_datetime(cur_date_int, unit='ns')
                idxh_date = self.dfh0.index[kidx_sets[0][-1]]

                # Gold in MT5 only valid in weekday 0~4
                if idx_date.weekday() >= 5:  # Saturday=5, Sunday=6
                    return new_orders, cancel_order_ids

                # If no position now, check if opening position
                if not cur_positions:
                    # check opening position
                    side = self.check_open(idxh_date)

                    ### if condition met, open order
                    if side!=0:
                        self.open_side = side
                        self.bar_cnt = 0

                        # Calculate base
                        base0, base1, broken = self.base_calc(cur_balance)

                        if not broken:
                            base0 = base0 if side==1 else -base0
                            base1 = -base1 if side==1 else base1
                            new_orders += [dict(symbol_idx = 0, position_id = 0, order_id = 0, base = base0, tag='open 0')]
                            new_orders += [dict(symbol_idx = 1, position_id = 1, order_id = 1, base = base1, tag='open 1')]

                # If having position now, check if closing position
                else:
                    # Acc bar
                    self.bar_cnt += 1

                    # check closing position
                    close_met = self.check_close(idx_date, idxh_date)

                    ### if condition met, add closing order
                    if close_met!=0:
                        cancel_order_ids = [order['order_id'] for order in pending_orders[0]] # Close S&P pending order
                        new_orders += [dict(symbol_idx = 0, position_id = 0, order_id = 2, tag='close 0')]
                        new_orders += [dict(symbol_idx = 1, position_id = 1, order_id = 3, tag='close 1')]

                        ### if closed, recheck open
                        side = self.check_open(idxh_date)

                        ### if condition met, open order
                        if side!=0:
                            self.open_side = side
                            self.bar_cnt = 0

                            # Calculate base
                            base0, base1, broken = self.base_calc(cur_balance)

                            if not broken:
                                base0 = base0 if side==1 else -base0
                                base1 = -base1 if side==1 else base1
                                new_orders += [dict(symbol_idx = 0, position_id = 0, order_id = 0, base = base0, tag='open 0')]
                                new_orders += [dict(symbol_idx = 1, position_id = 1, order_id = 1, base = base1, tag='open 1')]

                # Log print for kidx change
                self.log_print(f"INFO [{self.agent_name}]: {idx_date.strftime('%Y/%m/%d %H:%M:%S')}, bid0={self.bid0:.2f}, bid1={self.bid1:.2f}, spr={self.bid0 - self.bid1:.2f}, idxh_date={idxh_date.strftime('%Y/%m/%d %H:%M:%S')}, bol_l={self.spr_entry_l[idxh_date]:.2f}, bol_h={self.spr_entry_h[idxh_date]:.2f}", to_tg=0)

        return new_orders, cancel_order_ids


    def check_open(self, idxh):
        cur_spr = self.bid0 - self.bid1

        ### Check long condition
        long_condition = []
        if self.SIDE==0 or self.SIDE==1:
            # Pause
            long_condition.append(self.PAUSE==0)

            # BB met
            long_condition.append(cur_spr<self.spr_entry_l[idxh])

            if all(long_condition):
                return 1

        ### Check short condition
        short_condition = []
        if self.SIDE==0 or self.SIDE==2:
            # Pause
            short_condition.append(self.PAUSE==0)

            # BB met
            short_condition.append(cur_spr>self.spr_entry_h[idxh])

            if all(short_condition):
                return -1

        return 0


    def check_close(self, idx, idxh):
        # Org out
        cur_spr = self.bid0 - self.bid1
        close_condition = []
        if self.open_side==1:
            close_condition.append(cur_spr>self.spr_middle[idxh])

            if all(close_condition):
                return 1

        else:
            close_condition.append(cur_spr<self.spr_middle[idxh])

            if all(close_condition):
                return 1

        # Bar hour out
        if self.HOUR_OUT!=0:
            if self.bar_cnt>=60*self.HOUR_OUT:
                return 1

        return 0


    def base_calc(self, cur_balance):
        broken = False
        open_price = self.bid0
        quote = self.LEVERAGE * cur_balance / 2   # Divide by 2 since 2 symbols
        base = quote / open_price

        # base 1 calc
        base1 = (base // self.BASE_STEPS[1]) * self.BASE_STEPS[1] if self.BASE_STEPS[1]!=0 else base
        quote_act1 = base1 * open_price
        # Check broken
        if abs(base1)<=self.MIN_BASES[1]-(1e-7) or quote_act1<=self.MIN_QUOTES[1]:
            broken = True

        # base 0 calc
        # base0 = (base1 // self.BASE_STEPS[0]) * self.BASE_STEPS[0] if self.BASE_STEPS[0]!=0 else base1
        # quote_act0 = base0 * open_price
        # # Check broken
        # if abs(base0)<=self.MIN_BASES[0]-(1e-7) or quote_act0<=self.MIN_QUOTES[0]:
        #     broken = True
        base0 = base1


        return base0, base1, broken
