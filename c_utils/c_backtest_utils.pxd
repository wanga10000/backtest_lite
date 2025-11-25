# cython: language_level=3
# distutils: language=c++
import numpy as np
cimport numpy as np
from libc.stdint cimport int64_t


cdef class TickGen:
    cdef:
        int REAL_TICK
        int NO_COST
        bint HAS_TICK_TF
        np.float64_t[:] DEFAULT_SPREADS
        np.float64_t[:] SPREAD_FACTORS
        np.int64_t[:] SPREAD_POSES
        int[:] n_symbols_range
        int chunk_size
        np.float64_t[:, :] bids
        np.float64_t[:, :] asks
        np.int64_t[:, :] dates_int
        np.int64_t[:] chunk_number
        np.int64_t[:] max_chunks
        np.int64_t[:] ridx
        np.int64_t[:] max_rows
        np.int64_t[:] next_tick_time
        list fake_tick_df_filepath_list
        list tick_df_sets
        np.int64_t[:] start_idx
        np.int64_t[:] stop_idx
        int total_tick_num
        list fake_date_delta1
        list fake_date_delta2
        list fake_date_delta3

    cdef void _load_next_chunk(self, int sidx)
    cdef tuple get_tick(self)


cdef class kidxTracker:
    cdef:
        int has_df
        np.int64_t[:, :, :] date_sets  # [symbol_idx, tf_idx, date]
        np.int64_t[:, :] kidx_sets     # [symbol_idx, tf_idx]
        np.int64_t[:, :] date_len_sets # [symbol_idx, tf_idx]
        list kidx_sets_out # [symbol_idx, tf_idx]
        list kidx_changed_flags # [symbol_idx, tf_idx]
        int[:] n_tf_range
        int[:] n_symbols_range

    cdef void update_kidx(self, int sidx, np.int64_t cur_date_int)


cdef class EQTracker:
    cdef:
        int chunk_size
        double[:] balance_array
        int64_t[:] date_array
        int cur_idx
        bint first
        int64_t interval_ns
        int64_t next_date_int
        object eq_series

    cdef void _flush_to_series(self)
    cdef void update_balance(self, double cur_balance, int64_t cur_date_int)
    cpdef object get_eq(self)
