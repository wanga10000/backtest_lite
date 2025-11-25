import numpy as np
import pandas as pd
import math
from ta.volatility import AverageTrueRange
from ta.trend import EMAIndicator, SMAIndicator
import matplotlib.dates as mpl_dates
from datetime import timedelta


def heikin_ashi_k(df):  # return a new ha dataframe
    df_ha = df.copy(deep=True)
    df_ha['Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    df_ha['Open'] = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
    df_ha['High'] = df_ha[['Open', 'Close']].join(df['High']).max(axis=1)
    df_ha['Low'] = df_ha[['Open', 'Close']].join(df['Low']).min(axis=1)
    return df_ha


def supertrend(high, low, close, atr_len, atr_factor):
    # HL2 is simply the average of high and low prices
    hl2 = (high + low) / 2
    atr = AverageTrueRange(high=high, low=low, close=close, window=atr_len).average_true_range()

    # upperband and lowerband calculation
    # notice that final bands are set to be equal to the respective bands
    upperband = hl2 + (atr_factor * atr)
    lowerband = hl2 - (atr_factor * atr)

    # initialize Supertrend column to True
    supertrend = [1] * len(high)
    for i in range(1, len(high)):
        # if current close price crosses above upperband
        if close[i] > upperband[i - 1]:
            supertrend[i] = 1
        # if current close price crosses below lowerband
        elif close[i] < lowerband[i - 1]:
            supertrend[i] = -1
        # else, the trend continues
        else:
            supertrend[i] = supertrend[i - 1]
            # adjustment to the final bands
            if supertrend[i] == 1 and lowerband[i] < lowerband[i - 1]:
                lowerband[i] = lowerband[i - 1]
            if supertrend[i] == -1 and upperband[i] > upperband[i - 1]:
                upperband[i] = upperband[i - 1]
        # remove bands depending on the trend direction for visualization
        if supertrend[i] == 1:
            upperband[i] = np.nan
        else:
            lowerband[i] = np.nan

    upperband = np.nan_to_num(upperband)
    lowerband = np.nan_to_num(lowerband)

    band = upperband
    band[1:] = upperband[1:] + lowerband[1:]
    return atr, band, supertrend


def ha_supertrend(df, atr_len, atr_factor):
    # Make heikin ashi k
    df_ha = heikin_ashi_k(df)
    high = df_ha['High']
    low = df_ha['Low']
    close = df_ha['Close']

    # HL2 is changed to close due to ha
    hl2 = close

    atr = AverageTrueRange(high=high, low=low, close=close, window=atr_len).average_true_range()

    # upperband and lowerband calculation
    # notice that final bands are set to be equal to the respective bands
    upperband = hl2 + (atr_factor * atr)
    lowerband = hl2 - (atr_factor * atr)

    # initialize Supertrend column to True
    supertrend = [1] * len(high)
    for i in range(1, len(high)):
        # if current close price crosses above upperband
        if close[i] > upperband[i - 1]:
            supertrend[i] = 1
        # if current close price crosses below lowerband
        elif close[i] < lowerband[i - 1]:
            supertrend[i] = -1
        # else, the trend continues
        else:
            supertrend[i] = supertrend[i - 1]
            # adjustment to the final bands
            if supertrend[i] == 1 and lowerband[i] < lowerband[i - 1]:
                lowerband[i] = lowerband[i - 1]
            if supertrend[i] == -1 and upperband[i] > upperband[i - 1]:
                upperband[i] = upperband[i - 1]
        # remove bands depending on the trend direction for visualization
        if supertrend[i] == 1:
            upperband[i] = np.nan
        else:
            lowerband[i] = np.nan

    upperband = np.nan_to_num(upperband)
    lowerband = np.nan_to_num(lowerband)

    band = upperband
    band[1:] = upperband[1:] + lowerband[1:]
    return atr, band, supertrend


def demark_parallel(close, demark_len, lookback):
    close_shift = close.shift(lookback)
    close_compare_up = (close > close_shift)*1
    close_compare_up_sum = close_compare_up.rolling(demark_len).sum()
    close_compare_down = (close < close_shift)*1
    close_compare_down_sum = close_compare_down.rolling(demark_len).sum()
    dsi = (close_compare_up_sum==demark_len)*1 + (close_compare_down_sum==demark_len)*-1
    dsi_shift = dsi.shift(1)
    demark_up = ((dsi==-1) & (dsi_shift==0))*1
    demark_down = ((dsi==1) & (dsi_shift==0))*1
    return demark_up, demark_down


def NATR_improved(high, low, close, window, nor_window):
    tr0 = abs(high - low)
    tr1 = abs(high - close.shift())
    tr2 = abs(low - close.shift())
    tr = pd.Series(np.maximum.reduce([tr0,tr1,tr2]), index = tr0.index)
    tr_mean = tr.rolling(window).mean()
    tr_std = tr.rolling(window).std()
    tr_nor = (tr-tr_mean)/tr_std
    natr = EMAIndicator(close=tr_nor,
                        window=nor_window).ema_indicator()

    return natr


def efficiency_ratio(close, er_len):
    direction = close.diff(er_len).abs()
    volatility = close.diff().abs().rolling(er_len).sum()

    return direction / volatility


def rvol_by_time(df_volume, day_len, tf):
    date_series = df_volume.index
    volume_mean = df_volume.copy()
    volume_mean[:] = 0
    step = 30
    for t in range(0, 1440, step):
        hour = t // 60
        minute = t % 60
        hour_idx = date_series.hour == hour
        minute_idx = date_series.minute == minute
        df_moment = df_volume[hour_idx&minute_idx].copy()
        df_moment = df_moment.shift(1).rolling(day_len).mean()
        volume_mean.update(df_moment)

    rvol = df_volume/volume_mean
    rvol.replace(np.inf, np.nan, inplace=True)
    return rvol


def timeh_ind(date_series):
    def apply_hour(x):
        output = x.hour
        return output
    def apply_minute(x):
        output = x.minute
        return output

    # Calculate time float by hour
    timeh = np.array(list(map(apply_hour, date_series)))
    timem = np.array(list(map(apply_minute, date_series)))

    return timeh + timem/60


def timed_ind(date_series):
    def apply_day(x):
        output = x.weekday()
        return output

    # Calculate time float by day
    timed = np.array(list(map(apply_day, date_series)))

    return timed


def caseyc_ind(close, window, smooth):
    cchange = (close - close.shift(1))/close.shift(1)
    crank = (cchange - cchange.rolling(window).min()) / (cchange.rolling(window).max() - cchange.rolling(window).min()) * 100
    caseyc = SMAIndicator(close=crank, window = smooth).sma_indicator()
    return caseyc


def yuan_vola_regime(high, low, close, window=50):
    atrf = AverageTrueRange(high=high,
                            low=low,
                            close=close,
                            window=window).average_true_range()

    atrm = AverageTrueRange(high=high,
                            low=low,
                            close=close,
                            window=window*2).average_true_range()

    atrs = AverageTrueRange(high=high,
                            low=low,
                            close=close,
                            window=window*4).average_true_range()

    df = pd.DataFrame({'s': atrs, 'm': atrm, 'f': atrf})

    vo_reg = np.where((df['f'] < df['m']) & (df['m'] < df['s']), -1,
             np.where((df['f'] > df['m']) & (df['m'] > df['s']), 1, 0))


    return vo_reg


def yuan_trend_regime(close, open, window=50):
    a1 = 2 / (window + 1)
    u1_1 = np.zeros(len(close))
    u1_2 = np.zeros(len(close))
    d1_1 = np.zeros(len(close))
    d1_2 = np.zeros(len(close))

    for i in range(len(close)):
        if i == 0:
            u1_1[i] = close[i]
            u1_2[i] = close[i] ** 2
            d1_1[i] = close[i]
            d1_2[i] = close[i] ** 2
        u1_1[i] = max(close[i], open[i], u1_1[i-1] - (u1_1[i-1] - close[i]) * a1)
        u1_2[i] = max(close[i] ** 2, open[i] ** 2, u1_2[i-1] - (u1_2[i-1] - close[i] ** 2) * a1)
        d1_1[i] = min(close[i], open[i], d1_1[i-1] + (close[i] - d1_1[i-1]) * a1)
        d1_2[i] = min(close[i] ** 2, open[i] ** 2, d1_2[i-1] + (close[i] ** 2 - d1_2[i-1]) * a1)

    u1_1 = pd.Series(u1_1, index=close.index)
    u1_2 = pd.Series(u1_2, index=close.index)
    d1_1 = pd.Series(d1_1, index=close.index)
    d1_2 = pd.Series(d1_2, index=close.index)

    # Components
    bl_diff = d1_2 - d1_1 ** 2
    bl = bl_diff.apply(np.sqrt)
    br_diff = u1_2 - u1_1 ** 2
    br = br_diff.apply(np.sqrt)
    tr_reg = ((100 * (bl - br) / br) > 90)*1 + ((100 * (bl - br) / br) < -90)*(-1)

    return tr_reg


def tradj_ema(close, high, low, window, factor):
    Mltp1 = 2.0 / (window + 1.0)
    tr = pd.DataFrame(index=close.index)
    tr['tr0'] = abs(high - low)
    tr['tr1'] = abs(high - close.shift())
    tr['tr2'] = abs(low - close.shift())
    tr = tr[['tr0', 'tr1', 'tr2']].max(axis=1)
    HHV = tr.rolling(window=window).max()
    LLV = tr.rolling(window=window).min()
    TRAdj = (tr - LLV) / (HHV - LLV)
    Mltp2 = TRAdj * factor
    Rate = Mltp1 * (1.0 + Mltp2)
    TRAdjEMA = pd.Series(index=close.index)
    for i in range(len(close)):
        if i > window:
            prev = TRAdjEMA.iloc[i-1] if ~np.isnan(TRAdjEMA.iloc[i-1]) else close.iloc[i-1]
            TRAdjEMA.iloc[i] = prev + Rate.iloc[i] * (close.iloc[i] - prev)
        else:
            TRAdjEMA.iloc[i] = pd.NA
    return TRAdjEMA


def sd_atr(df, window):
    atr = AverageTrueRange(high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            window=window).average_true_range()

    sd = df['Close'].rolling(window).std()
    return sd - atr


class multi_symbols_synchronizer():
    def __init__(self, symbol_num):
        self.states = [0] * symbol_num
        self.dates = [0] * symbol_num


    def act(self, cur_sidx, cur_date_int):
        # Update state and datetime for the triggered cur_sidx
        self.states[cur_sidx] = 1
        self.dates[cur_sidx] = pd.to_datetime(cur_date_int)

        # Check if all states are activated
        if all(s == 1 for s in self.states):
            # Check if all datetime matches without second
            normalized_dates = [(dt.year, dt.month, dt.day, dt.hour, dt.minute) for dt in self.dates]
            self.states = [0] * len(self.states)  # Reset all states

            if len(set(normalized_dates)) == 1:
                return 1
            else:
                # If not, update cur_sidx state to 1 and return 0
                self.states[cur_sidx] = 1
                self.dates[cur_sidx] = pd.to_datetime(cur_date_int)
                return 0
        else:
            return 0