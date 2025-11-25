# cython: language_level=3
# distutils: language=c++
import numpy as np
import pandas as pd
from utils.utils_general import get_rightmost_time_idx, get_leftmost_time_idx
import collections
cimport numpy as np
np.import_array()
import time
from libc.stdint cimport int64_t


cdef class TickGen:
    def __init__(self, agent_dict):
        symbols = str(agent_dict['general']['SYMBOLS']).split(",")
        n_symbols = len(symbols)
        self.n_symbols_range = np.arange(1, n_symbols, dtype=np.int32)
        self.REAL_TICK = int(agent_dict['backtest']['REAL_TICK'])
        self.NO_COST = int(agent_dict['backtest']['NO_COST'])
        self.HAS_TICK_TF = 'tick_df_sets' in agent_dict.keys()
        START_DATE = agent_dict['backtest']['START_DATE']
        STOP_DATE = agent_dict['backtest']['STOP_DATE']

        # Symbol duplicated check
        if [item for item, count in collections.Counter(symbols).items() if count > 1]:
            raise ValueError("Duplicate symbols found in SYMBOLS")

        if self.REAL_TICK==1 and not self.HAS_TICK_TF:
            raise ValueError("REAL_TICK=1 mode not supported yet")
        else:
            # Initialize
            self.chunk_size = 500000
            self.bids = np.zeros((n_symbols, self.chunk_size*4), dtype=np.float64)
            self.asks = np.zeros((n_symbols, self.chunk_size*4), dtype=np.float64)
            self.dates_int = np.zeros((n_symbols, self.chunk_size*4), dtype=np.int64)
            self.ridx = np.zeros(n_symbols, dtype=np.int64)
            self.max_rows = np.zeros(n_symbols, dtype=np.int64)
            self.chunk_number = np.zeros(n_symbols, dtype=np.int64)
            self.max_chunks = np.zeros(n_symbols, dtype=np.int64)
            self.next_tick_time = np.zeros(n_symbols, dtype=np.int64)
            self.start_idx = np.zeros(n_symbols, dtype=np.int64)
            self.stop_idx = np.zeros(n_symbols, dtype=np.int64)
            self.DEFAULT_SPREADS = np.zeros(n_symbols, dtype=np.float64)
            self.SPREAD_FACTORS = np.zeros(n_symbols, dtype=np.float64)
            self.SPREAD_POSES = np.zeros(n_symbols, dtype=np.int64)
            self.fake_tick_df_filepath_list = []
            self.fake_date_delta1 = []
            self.fake_date_delta2 = []
            self.fake_date_delta3 = []
            self.total_tick_num = 0

            # Get tick tf for each symbol
            fake_tick_tfs = str(agent_dict['backtest']['FAKE_TICK_TF']).split(",")
            if len(fake_tick_tfs) == 1: # If only one tf is provided, use it for all symbols
                fake_tick_tfs = fake_tick_tfs * n_symbols

            # Set stuff for all symbols
            for sidx, _ in enumerate(symbols):
                # Set spread
                symbol_info = agent_dict['symbols_info'][sidx]
                self.SPREAD_POSES[sidx] = int(symbol_info['SPREAD_POS'])
                self.SPREAD_FACTORS[sidx] = float(symbol_info['SPREAD_FACTOR'])
                self.DEFAULT_SPREADS[sidx] = float(symbol_info['DEFAULT_SPREAD'])

                # Set fake_date_delta
                fake_tick_tf = fake_tick_tfs[sidx]
                tf_value = int(''.join(filter(str.isdigit, fake_tick_tf)))
                tf_unit = ''.join(filter(str.isalpha, fake_tick_tf))
                seconds_per_bar = (tf_value * 60 if tf_unit == 'm' else
                                tf_value * 3600 if tf_unit == 'h' else
                                tf_value * 86400)

                self.fake_date_delta1.append(pd.Timedelta(seconds=seconds_per_bar//3))
                self.fake_date_delta2.append(pd.Timedelta(seconds=2*seconds_per_bar//3))
                self.fake_date_delta3.append(pd.Timedelta(seconds=seconds_per_bar-1))

                # Read full df and get start and stop idx
                if not self.HAS_TICK_TF:
                    # Set fake tick df filepath
                    fake_tick_df_filepath = symbol_info['DATA_PATH'] + '/' + fake_tick_tf + '.parquet'
                    self.fake_tick_df_filepath_list.append(fake_tick_df_filepath)
                    df = pd.read_parquet(fake_tick_df_filepath)
                else:
                    self.tick_df_sets = agent_dict['tick_df_sets']
                    df = self.tick_df_sets[sidx]
                    
                df_date = df['Date'].values.squeeze()
                self.start_idx[sidx] = get_rightmost_time_idx(np.datetime64(START_DATE), df_date)
                self.stop_idx[sidx] = get_leftmost_time_idx(np.datetime64(STOP_DATE), df_date) + 1
                self.total_tick_num += len(df.iloc[self.start_idx[sidx]:self.stop_idx[sidx], :]) * 4
                self.max_chunks[sidx] = np.ceil(len(df.iloc[self.start_idx[sidx]:self.stop_idx[sidx], :]) / self.chunk_size) - 1

                # Load first chunk
                self._load_next_chunk(sidx)


    cdef void _load_next_chunk(self, int sidx):
        if self.REAL_TICK==1:
            pass
        else:
            # Get start and over row idx
            start_row = self.start_idx[sidx] + self.chunk_number[sidx] * self.chunk_size
            end_row = min(start_row + self.chunk_size, self.stop_idx[sidx])

            # read df chunk
            if not self.HAS_TICK_TF:
                df = pd.read_parquet(self.fake_tick_df_filepath_list[sidx]).iloc[start_row:end_row, :]
            else:
                df = self.tick_df_sets[sidx].iloc[start_row:end_row, :].copy()

            # df->ticks (dates, bids, asks)
            df_len = len(df['Open'].values)
            is_uptrend = df['Close'].values > df['Open'].values
            indices = np.arange(df_len) * 4

            bids = np.asarray(self.bids)
            bids[sidx, indices] = df['Open'].values
            bids[sidx, indices + 1] = np.where(is_uptrend, df['Low'].values, df['High'].values)
            bids[sidx, indices + 2] = np.where(is_uptrend, df['High'].values, df['Low'].values)
            bids[sidx, indices + 3] = df['Close'].values
            self.bids = bids

            asks = np.asarray(self.asks)
            spreads_arr = np.zeros(self.chunk_size, dtype=np.float64)
            df['Spread'] = df['Spread']*self.SPREAD_FACTORS[sidx] if 'Spread' in df else self.DEFAULT_SPREADS[sidx]
            df['Spread'] = df['Spread'] * (1/(10**self.SPREAD_POSES[sidx]))
            df['Spread'] = 0 if self.NO_COST==1 else df['Spread']
            spreads_arr[:df_len] = df['Spread'].values.astype(np.float64)
            asks[sidx, :] = bids[sidx, :] + np.repeat(spreads_arr, 4).astype(np.float64)
            self.asks = asks

            dates_int = np.asarray(self.dates_int)
            dates = np.empty(self.chunk_size * 4, dtype='datetime64[ns]')
            dates[indices] = df['Date'].values
            dates[indices + 1] = df['Date'].values + self.fake_date_delta1[sidx]
            dates[indices + 2] = df['Date'].values + self.fake_date_delta2[sidx]
            dates[indices + 3] = df['Date'].values + self.fake_date_delta3[sidx]
            dates_int[sidx, :] = dates.astype('int64')
            self.dates_int = dates_int

            # Reset current row and total rows
            self.ridx[sidx] = 0
            self.max_rows[sidx] = df_len * 4 - 1
            self.chunk_number[sidx] += 1
            self.next_tick_time[sidx] = self.dates_int[sidx, 0]

    cdef tuple get_tick(self):
        cdef:
            int sidx
            int min_sidx = 0
            np.int64_t min_time = self.next_tick_time[0]

        # find the symbol with the smallest next_tick_time
        for sidx in self.n_symbols_range:
            if self.next_tick_time[sidx] < min_time:
                min_time = self.next_tick_time[sidx]
                min_sidx = sidx
        
        cdef:
            np.float64_t bid = self.bids[min_sidx, self.ridx[min_sidx]]
            np.float64_t ask = self.asks[min_sidx, self.ridx[min_sidx]]
            np.int64_t cur_date_int = self.dates_int[min_sidx, self.ridx[min_sidx]]
        
        # If no tick left, set a big time and return advance and don't update ridx
        if self.chunk_number[min_sidx] > self.max_chunks[min_sidx] and self.ridx[min_sidx] == self.max_rows[min_sidx]:
            self.next_tick_time[min_sidx] = 2534021007999999999
            return min_sidx, cur_date_int, bid, ask

        # Acc ridx and check if load next chunk
        self.ridx[min_sidx] = self.ridx[min_sidx] + 1

        if self.ridx[min_sidx] > self.max_rows[min_sidx]:
            self._load_next_chunk(min_sidx)
            
        # Update next_tick_time
        self.next_tick_time[min_sidx] = self.dates_int[min_sidx, self.ridx[min_sidx]]

        return min_sidx, cur_date_int, bid, ask


cdef class kidxTracker:
    def __init__(self, list df_sets):
        # Check if no df required, tag has_df=0 and return
        self.has_df = 1
        if not df_sets:
            self.has_df = 0
            return

        # Initialize
        n_symbols = len(df_sets)
        n_tf = len(df_sets[0])
        self.n_symbols_range = np.arange(n_symbols, dtype=np.int32)
        self.n_tf_range = np.arange(n_tf, dtype=np.int32)

        max_len = max(len(df) for df_set in df_sets for df in df_set)
        date_sets = np.zeros((n_symbols, n_tf, max_len), dtype=np.int64)
        self.kidx_sets = np.ones((n_symbols, n_tf), dtype=np.int64)
        self.date_len_sets = np.zeros((n_symbols, n_tf), dtype=np.int64)
        self.kidx_sets_out = [[0] * n_tf for _ in range(n_symbols)]
        self.kidx_changed_flags = [[0] * n_tf for _ in range(n_symbols)]

        for sidx, df_set in enumerate(df_sets):
            for tf_idx, df in enumerate(df_set):
                dates = df.index.values.astype(np.int64)
                df_len = len(dates)
                self.date_len_sets[sidx, tf_idx] = df_len
                date_sets[sidx, tf_idx, :df_len] = dates

        self.date_sets = date_sets

    cdef void update_kidx(self, int sidx, np.int64_t cur_date_int):
        if self.has_df==0: return

        cdef:
            int tf_idx
            np.int64_t kidx
            np.int64_t date_len
            np.int64_t nxt_date

        for ssidx in self.n_symbols_range:
            if ssidx==sidx:
                for tf_idx in self.n_tf_range:
                    kidx_list = self.kidx_sets_out[sidx]
                    changed_flags = self.kidx_changed_flags[sidx]
                    kidx = self.kidx_sets[sidx, tf_idx]
                    date_len = self.date_len_sets[sidx, tf_idx]
                    changed_flags[tf_idx] = 0

                    while kidx < date_len - 1:
                        nxt_date = self.date_sets[sidx, tf_idx, kidx + 1]
                        if cur_date_int >= nxt_date:
                            kidx_list[tf_idx] = kidx
                            changed_flags[tf_idx] = 1
                            kidx += 1
                        else:
                            break

                    self.kidx_sets[sidx, tf_idx] = kidx
            else:
                for tf_idx in self.n_tf_range:
                    self.kidx_changed_flags[ssidx][tf_idx] = 0


cdef class EQTracker:
    def __init__(self, str eq_tf):
        # Initialize arrays
        self.chunk_size = 500000
        self.balance_array = np.zeros(self.chunk_size, dtype=np.float64)
        self.date_array = np.zeros(self.chunk_size, dtype=np.int64)
        self.cur_idx = 0
        
        # Convert eq_tf to pandas format
        cdef int64_t tf_value = int(''.join(filter(str.isdigit, eq_tf)))
        cdef str tf_unit = ''.join(filter(str.isalpha, eq_tf))
        
        if tf_unit == 'm':
            self.interval_ns = tf_value * np.int64(60) * np.int64(1_000_000_000)
        elif tf_unit == 'h':
            self.interval_ns = tf_value * np.int64(3600) * np.int64(1_000_000_000)
        elif tf_unit == 'd':
            self.interval_ns = tf_value * np.int64(86400) * np.int64(1_000_000_000)
        else:
            raise ValueError("Invalid eq_tf unit. Use 'm', 'h' or 'd'")
            
        self.eq_series = pd.Series(dtype=np.float64)
        self.next_date_int = 0
        self.first = True
        
    cdef void _flush_to_series(self):
        if self.cur_idx > 0:
            temp_series = pd.Series(np.asarray(self.balance_array[:self.cur_idx]), index=np.asarray(self.date_array[:self.cur_idx]))
            self.eq_series = pd.concat([self.eq_series, temp_series])
            self.cur_idx = 0
            
    cdef void update_balance(self, double cur_balance, int64_t cur_date_int):
        if self.first:
            self.next_date_int = (cur_date_int // self.interval_ns) * self.interval_ns
            self.first = False

        if cur_date_int >= self.next_date_int:
            while cur_date_int >= self.next_date_int:
                self.date_array[self.cur_idx] = self.next_date_int
                self.next_date_int += self.interval_ns

            self.balance_array[self.cur_idx] = cur_balance
            self.cur_idx += 1

            if self.cur_idx >= self.chunk_size:
                self._flush_to_series()

    cpdef object get_eq(self):
        # Final flush and return complete series
        self._flush_to_series()
        self.eq_series.index = pd.to_datetime(self.eq_series.index)
        return self.eq_series.fillna(method='ffill')