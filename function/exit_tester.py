import numpy as np
import pandas as pd
from ta.volatility import AverageTrueRange
from ta.trend import EMAIndicator, CCIIndicator, AroonIndicator, ADXIndicator
from function.func_lib import supertrend, NATR_improved, efficiency_ratio


class exit_tester:
    def __init__(self):
        pass


    def load_strategy_cfg(self, strategy_cfg, USED_TFS):
        self.USED_TFS = USED_TFS
        self.main_tf_idx = USED_TFS.index(str(strategy_cfg['MAIN_TF']))
        self.EXITER_TYPES = list(map(int, str(strategy_cfg['EXITER_TYPES']).split(",")))
        self.EXITER_PER_STOP_FAC = float(strategy_cfg['EXITER_PER_STOP_FAC'])
        self.EXITER_PER_PROFIT_FAC = float(strategy_cfg['EXITER_PER_PROFIT_FAC'])
        self.EXITER_ATR_LEN = int(strategy_cfg['EXITER_ATR_LEN'])
        self.EXITER_ATR_STOP_FAC = float(strategy_cfg['EXITER_ATR_STOP_FAC'])
        self.EXITER_ATR_PROFIT_FAC = float(strategy_cfg['EXITER_ATR_PROFIT_FAC'])
        self.EXITER_PER_TRAIL_STOP_FAC = float(strategy_cfg['EXITER_PER_TRAIL_STOP_FAC'])
        self.EXITER_ATR_TRAIL_STOP_FAC = float(strategy_cfg['EXITER_ATR_TRAIL_STOP_FAC'])
        self.EXITER_BAR_CNT = int(strategy_cfg['EXITER_BAR_CNT'])
        self.EXITER_WEEKDAY = int(strategy_cfg['EXITER_WEEKDAY'])


    def load_df_sets(self, df_sets):
        self.df_sets = df_sets
        self.declare_df()


    def declare_df(self):
        # Main symbol df handle
        df_set = self.df_sets[0]

        # Main tf handle
        self.df = df_set[self.main_tf_idx]


    def indicator_calc(self):
        # For E0~E4, suggest only using it with default exit in original strategy. Don't use independently.
        # E0.  No exit
        # E1.  % stoploss
        # E2.  % profit
        # E3.  ATR stoploss
        # E4.  ATR profit
        # E5.  % trailing stop
        # E6.  ATR trailing stop
        # E7.  N bars out
        # E8.  N bars loss out
        # E9.  N bars win out
        # E10. End of day (close at every 23:56)
        # E11. End of week (close at every Friday 23:56)
        for exit_type in self.EXITER_TYPES:
            if exit_type in [3, 4, 6]:
                self.df['EXIT_ATR'] = AverageTrueRange(high=self.df['High'],
                                                       low=self.df['Low'],
                                                       close=self.df['Close'],
                                                       window=self.EXITER_ATR_LEN).average_true_range()
            else:
                pass


    def open_reset(self, open_side, baseline_price, kidx_sets):
        # E0.  No exit
        # E1.  % stoploss
        # E2.  % profit
        # E3.  ATR stoploss
        # E4.  ATR profit
        # E5.  % trailing stop
        # E6.  ATR trailing stop
        # E7.  N bars out
        # E8.  N bars loss out
        # E9.  N bars win out
        # E10. End of day (close at every 23:56)
        # E11. End of week (close at every Friday 23:56)
        self.open_side = open_side
        self.baseline_price = baseline_price

        if 5 in self.EXITER_TYPES:
            self.trail_stoploss = baseline_price * (1 - open_side * (self.EXITER_PER_TRAIL_STOP_FAC/100))

        if 6 in self.EXITER_TYPES:
            idx = kidx_sets[0][self.main_tf_idx]
            self.trail_stoploss = baseline_price - open_side * self.EXITER_ATR_TRAIL_STOP_FAC * self.df['EXIT_ATR'][idx]

        if any(x in self.EXITER_TYPES for x in [7, 8, 9]):
            self.bar_cnt = 0


    def check_close(self, cur_date_int, bid, ask, kidx_sets, kidx_changed_flags):
        conditions = []

        for exit_type in self.EXITER_TYPES:
            # E0.  No exit
            # E1.  % stoploss
            # E2.  % profit
            # E3.  ATR stoploss
            # E4.  ATR profit
            # E5.  % trailing stop
            # E6.  ATR trailing stop
            # E7.  N bars out
            # E8.  N bars loss out
            # E9.  N bars win out
            # E10. End of day (close at every 23:56)
            # E11. End of week (close at every Friday 23:56)
            if exit_type==0:
                pass

            elif exit_type==1:
                cur_price = bid if self.open_side==1 else ask
                stoploss = self.baseline_price * (1 - self.open_side * (self.EXITER_PER_STOP_FAC/100))
                conditions.append(self.open_side * cur_price <= self.open_side * stoploss)

            elif exit_type==2:
                cur_price = bid if self.open_side==1 else ask
                profit = self.baseline_price * (1 + self.open_side * (self.EXITER_PER_PROFIT_FAC/100))
                conditions.append(self.open_side * cur_price >= self.open_side * profit)

            elif exit_type==3:
                idx = kidx_sets[0][self.main_tf_idx]
                cur_price = bid if self.open_side==1 else ask
                stoploss = self.baseline_price - self.open_side * self.EXITER_ATR_STOP_FAC * self.df['EXIT_ATR'][idx]
                conditions.append(self.open_side * cur_price <= self.open_side * stoploss)

            elif exit_type==4:
                idx = kidx_sets[0][self.main_tf_idx]
                cur_price = bid if self.open_side==1 else ask
                profit = self.baseline_price + self.open_side * self.EXITER_ATR_PROFIT_FAC * self.df['EXIT_ATR'][idx]
                conditions.append(self.open_side * cur_price >= self.open_side * profit)

            elif exit_type==5:
                cur_price = bid if self.open_side==1 else ask
                conditions.append(self.open_side * cur_price <= self.open_side * self.trail_stoploss)
                if self.open_side==1:
                    self.trail_stoploss = max(self.trail_stoploss, cur_price * (1 - (self.EXITER_PER_TRAIL_STOP_FAC/100)))
                else:
                    self.trail_stoploss = min(self.trail_stoploss, cur_price * (1 + (self.EXITER_PER_TRAIL_STOP_FAC/100)))

            elif exit_type==6:
                idx = kidx_sets[0][self.main_tf_idx]
                cur_price = bid if self.open_side==1 else ask
                conditions.append(self.open_side * cur_price <= self.open_side * self.trail_stoploss)
                if self.open_side==1:
                    self.trail_stoploss = max(self.trail_stoploss, cur_price - self.EXITER_ATR_TRAIL_STOP_FAC * self.df['EXIT_ATR'][idx])
                else:
                    self.trail_stoploss = min(self.trail_stoploss, cur_price + self.EXITER_ATR_TRAIL_STOP_FAC * self.df['EXIT_ATR'][idx])

            elif exit_type==7:
                if kidx_changed_flags[0][self.main_tf_idx]:
                    self.bar_cnt += 1
                    conditions.append(self.bar_cnt >= self.EXITER_BAR_CNT)

            elif exit_type==8:
                if kidx_changed_flags[0][self.main_tf_idx]:
                    self.bar_cnt += 1
                    cur_price = bid if self.open_side==1 else ask
                    local_cons = [(self.bar_cnt >= self.EXITER_BAR_CNT)]
                    local_cons.append(self.open_side*cur_price < self.open_side*self.baseline_price)
                    conditions.append(all(local_cons))

            elif exit_type==9:
                if kidx_changed_flags[0][self.main_tf_idx]:
                    self.bar_cnt += 1
                    cur_price = bid if self.open_side==1 else ask
                    local_cons = [(self.bar_cnt >= self.EXITER_BAR_CNT)]
                    local_cons.append(self.open_side*cur_price > self.open_side*self.baseline_price)
                    conditions.append(all(local_cons))

            elif exit_type==10:
                cur_date = pd.Timestamp(cur_date_int)
                conditions.append(cur_date.hour == 23 and (56 <= cur_date.minute < 59))

            elif exit_type==11:
                cur_date = pd.Timestamp(cur_date_int)
                conditions.append(cur_date.day_of_week==self.EXITER_WEEKDAY and cur_date.hour == 23 and (56 <= cur_date.minute < 59))

        return any(conditions)