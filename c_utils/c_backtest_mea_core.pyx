# cython: language_level=3
# distutils: language=c++
import glob, os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.rrule import rrule, HOURLY
from importlib import import_module
from utils.utils_general import read_df_sets, get_leftmost_time_idx, printProgressBar
from copy import deepcopy
cimport numpy as np
np.import_array()
from libc.stdint cimport int64_t
from libc.math cimport fabs, fmod
from c_utils.c_backtest_utils cimport TickGen, kidxTracker, EQTracker
from ta.volatility import AverageTrueRange # MEA
from decimal import Decimal


def dummy_log_print(msg, to_tg=0):
    # Dummy log_print function that does nothing
    pass


def cur_positions_change(agent_name, order, cur_positions, position_cnt, symbols_info, cur_base_balance, ask, bid, slippage, cur_date_int, mea_exurs_list, mea_dates_list, kidx_sets, atr_df, datr_df, atr_tf_idx, datr_tf_idx):
    if order['position_id'] in cur_positions.keys():
        # Check base is 0
        if order['base'] != 0:
            raise ValueError(f"ERROR [{agent_name}][cur_positions_change]: Can't add base on current position {order['position_id']} for order {order['order_id']} at {cur_date_int}")

        position_info = cur_positions.pop(order['position_id'])

        # Check sidx the same
        sidx = order['symbol_idx']
        if sidx != position_info['symbol_idx']:
            raise ValueError(f"ERROR [{agent_name}][cur_positions_change]: Can't close current position {order['position_id']} by different symbol for order {order['order_id']} at {cur_date_int}")

        # Close fee calculate: 'trade_fee', 'spread_fee', 'slippage_fee'
        symbol_info = symbols_info[sidx]
        TRADE_FEE_TYPE = symbol_info['TRADE_FEE_TYPE']
        order_type = order['order_type']
        base = -position_info['base']
        trade_fee_rate = symbol_info['MAKER_FEE'] if order_type==1 else symbol_info['TAKER_FEE']
        end_quote = abs(base) * order['exec_price']
        if TRADE_FEE_TYPE==0:
            trade_fee = end_quote * trade_fee_rate
        else: # TRADE_FEE_TYPE==1
            trade_fee = abs(base) * trade_fee_rate
        if base>0:
            spread = ask - bid
            spread_fee = base * spread
        else:
            spread_fee = 0
        slippage_fee = base * slippage  # If base<0, slippage is also <0

        # Close fee calculate: 'swap_fee'
        SWAP_TYPE = symbol_info['SWAP_SETTLE_HOUR']
        SWAP_SETTLE_HOUR = symbol_info['SWAP_SETTLE_HOUR']
        SWAP_SETTLE_DAY = symbol_info['SWAP_SETTLE_DAY']
        open_date = pd.Timestamp(position_info['open_date']).replace(minute=0, second=0, microsecond=0)
        open_date = (open_date + timedelta(hours=1)) if open_date.hour in SWAP_SETTLE_HOUR else open_date
        close_date = pd.Timestamp(order['exec_date']).replace(minute=0, second=0, microsecond=0)
        count = 0
        for date_time in rrule(freq=HOURLY, dtstart=open_date, until=close_date):
            if date_time.hour in SWAP_SETTLE_HOUR:
                count += SWAP_SETTLE_DAY[date_time.weekday()]

        swap = symbol_info['SHORT_SWAP'] if base>0 else symbol_info['LONG_SWAP'] # Open direction is the opposite of close direction
        if SWAP_TYPE==0:
            swap_fee = -swap * end_quote * count  # Estimated. If swap is positive, it makes money.
        else: # SWAP_TYPE==1
            swap_fee = -swap * abs(base) * count  # Estimated. If swap is positive, it makes money.

        # Update cur_base_balance & order
        pnl = -base * (order['exec_price'] - position_info['open_price']) - trade_fee - swap_fee - position_info['open_fee'] 
        mfe = pnl if pnl > position_info['mfe'] else position_info['mfe'] 
        mae = -pnl if -pnl > position_info['mae'] else position_info['mae']

        # MEA
        open_base = position_info['base']
        exur_price = order['exec_price'] - position_info['open_price'] if open_base>0 else position_info['open_price'] - order['exec_price']
        mea_pnl_price = pnl / abs(base)
        mea_exurs_list.append(exur_price)
        mea_dates_list.append(order['exec_date'])

        exurs_array = np.array(mea_exurs_list)
        max_index = np.argmax(exurs_array)
        min_index = np.argmin(exurs_array)
        mea_mfe_price = mea_exurs_list[max_index] if mea_exurs_list[max_index] > 0 else 0
        mea_mae_price = -mea_exurs_list[min_index]
        mea_mfe_date = mea_dates_list[max_index]
        mea_mae_date = mea_dates_list[min_index]

        if min_index == 0:
            mea_bmfe_price = 0
            mea_bmfe_date = mea_dates_list[0]
            mea_cmfe_price = mea_mfe_price
            mea_cmfe_date = mea_mfe_date
        elif min_index > 0 and min_index != (len(exurs_array)-1):
            mea_bmfe_array = exurs_array[:min_index]
            mea_bmfe_index = np.argmax(mea_bmfe_array)
            mea_bmfe_price = mea_exurs_list[mea_bmfe_index] if mea_exurs_list[mea_bmfe_index] > 0 else 0
            mea_bmfe_date = mea_dates_list[mea_bmfe_index]
            mea_cmfe_array = exurs_array[min_index+1:]
            mea_cmfe_index = np.argmax(mea_cmfe_array) + min_index + 1
            mea_cmfe_price = mea_exurs_list[mea_cmfe_index] if mea_exurs_list[mea_cmfe_index] > 0 else 0
            mea_cmfe_date = mea_dates_list[mea_cmfe_index]
        else: # min_index == (len(exurs_array)-1):
            mea_bmfe_array = exurs_array[:min_index]
            mea_bmfe_index = np.argmax(mea_bmfe_array)
            mea_bmfe_price = mea_exurs_list[mea_bmfe_index]
            mea_bmfe_date = mea_dates_list[mea_bmfe_index]
            mea_cmfe_price = 0
            mea_cmfe_date = order['exec_date']

        cumulative_max = np.maximum.accumulate(exurs_array)
        drawdowns = cumulative_max - exurs_array
        mea_mhl_price = np.max(drawdowns)

        # MEA per handle
        mea_mfe_per = mea_mfe_price / position_info['open_price']
        mea_bmfe_per = mea_bmfe_price / position_info['open_price']
        mea_cmfe_per = mea_cmfe_price / position_info['open_price']
        mea_mae_per = mea_mae_price / position_info['open_price']
        mea_pnl_per = mea_pnl_price / position_info['open_price']
        mea_mhl_per = mea_mhl_price / position_info['open_price']

        # MEA atr handle
        mea_mfe_atr = mea_mfe_price / position_info['open_atr']
        mea_bmfe_atr = mea_bmfe_price / position_info['open_atr']
        mea_cmfe_atr = mea_cmfe_price / position_info['open_atr']
        mea_mae_atr = mea_mae_price / position_info['open_atr']
        mea_pnl_atr = mea_pnl_price / position_info['open_atr']
        mea_mhl_atr = mea_mhl_price / position_info['open_atr']

        # MEA atr handle
        mea_mfe_datr = mea_mfe_price / position_info['open_datr']
        mea_bmfe_datr = mea_bmfe_price / position_info['open_datr']
        mea_cmfe_datr = mea_cmfe_price / position_info['open_datr']
        mea_mae_datr = mea_mae_price / position_info['open_datr']
        mea_pnl_datr = mea_pnl_price / position_info['open_datr']
        mea_mhl_datr = mea_mhl_price / position_info['open_datr']

        cur_base_balance += pnl
        order.update({'position_cnt': position_info['position_cnt'], 'exec_base': base, 'trade_fee': trade_fee,
                      'spread_fee': spread_fee, 'slippage_fee': slippage_fee, 'swap_fee': swap_fee, 'pnl': pnl, 'mfe': mfe, 'mae': mae,
                      'mea_mfe_price': mea_mfe_price, 'mea_bmfe_price': mea_bmfe_price, 'mea_cmfe_price': mea_cmfe_price, 'mea_mae_price': mea_mae_price, 'mea_pnl_price': mea_pnl_price, 'mea_mhl_price': mea_mhl_price,  # MEA
                      'mea_mfe_per': mea_mfe_per, 'mea_bmfe_per': mea_bmfe_per, 'mea_cmfe_per': mea_cmfe_per, 'mea_mae_per': mea_mae_per, 'mea_pnl_per': mea_pnl_per, 'mea_mhl_per': mea_mhl_per,  # MEA
                      'mea_mfe_atr': mea_mfe_atr, 'mea_bmfe_atr': mea_bmfe_atr, 'mea_cmfe_atr': mea_cmfe_atr, 'mea_mae_atr': mea_mae_atr, 'mea_pnl_atr': mea_pnl_atr, 'mea_mhl_atr': mea_mhl_atr,  # MEA
                      'mea_mfe_datr': mea_mfe_datr, 'mea_bmfe_datr': mea_bmfe_datr, 'mea_cmfe_datr': mea_cmfe_datr, 'mea_mae_datr': mea_mae_datr, 'mea_pnl_datr': mea_pnl_datr, 'mea_mhl_datr': mea_mhl_datr,  # MEA
                      'mea_mfe_date': mea_mfe_date, 'mea_bmfe_date': mea_bmfe_date, 'mea_cmfe_date': mea_cmfe_date, 'mea_mae_date': mea_mae_date}) # MEA

    else:
        # Check base is not 0
        if order['base'] == 0:
            raise ValueError(f"ERROR [{agent_name}][cur_positions_change]: Can't add 0 base on not existed position {order['position_id']} for order {order['order_id']} at {cur_date_int}")

        # Acc postion cnt
        position_cnt += 1

        # Open fee calculate: 'trade_fee', 'spread_fee', 'slippage_fee'
        symbol_info = symbols_info[order['symbol_idx']]
        TRADE_FEE_TYPE = symbol_info['TRADE_FEE_TYPE']
        order_type = order['order_type']
        base = order['base']
        trade_fee_rate = symbol_info['MAKER_FEE'] if order_type==1 else symbol_info['TAKER_FEE']
        initial_quote = abs(base) * order['exec_price']
        if TRADE_FEE_TYPE==0:
            trade_fee = initial_quote * trade_fee_rate
        else: # TRADE_FEE_TYPE==1
            trade_fee = abs(base) * trade_fee_rate
        if base>0:
            spread = ask - bid
            spread_fee = base * spread
        else:
            spread_fee = 0
        slippage_fee = base * slippage  # If base<0, slippage is also <0

        cur_positions[order['position_id']] = dict(position_cnt=position_cnt, symbol_idx=order['symbol_idx'], open_date=order['exec_date'], open_price=order['exec_price'],
                                                   base=base, open_fee=trade_fee, float_pnl=-trade_fee, mfe=0, mae=0,
                                                   open_atr=atr_df[kidx_sets[0][atr_tf_idx]], open_datr=datr_df[kidx_sets[0][datr_tf_idx]]) # MEA
        order.update({'position_cnt': position_cnt, 'exec_base': base, 'trade_fee': trade_fee,
                      'spread_fee': spread_fee, 'slippage_fee': slippage_fee, 'swap_fee': 0})

        # MEA
        mea_exurs_list = [0]
        mea_dates_list = [order['exec_date']]

    return order, cur_positions, position_cnt, cur_base_balance, mea_exurs_list, mea_dates_list


def check_market_order_valid(agent_name, has_base_limit, cur_positions, order, sidx, bid, ask, min_quote, min_base, base_step, cur_date_int):
    # 1. Check base valid for cur_position
    # 2. Check base valid for symbols_info rule
    pos_id = order['position_id']
    has_position = pos_id in cur_positions.keys()
    base = order['base']

    ### Check base valid for cur_position
    # Error 1
    if not has_position:
        side = order['base']>0
        if base==0:
            raise ValueError(f"ERROR [{agent_name}][check_order_valid]: Can't add 0 base on not existed position {pos_id} for order {order['order_id']} at {cur_date_int}")

    # Error 2 & 3
    else:
        pos_sidx = cur_positions[pos_id]['symbol_idx']
        side = cur_positions[pos_id]['base']<0
        if base!=0:
            raise ValueError(f"ERROR [{agent_name}][check_order_valid]: Can't add base on current position {pos_id} for order {order['order_id']} at {cur_date_int}")
        if sidx!=pos_sidx:
            raise ValueError(f"ERROR [{agent_name}][check_order_valid]: Can't close current position {pos_id} by different symbol for order {order['order_id']} at {cur_date_int}")

    ### Check base valid for symbols_info rule
    open_price = ask if side else bid
    if has_base_limit and not has_position:
        abs_base = abs(base)
        quote_value = abs_base * open_price
        if quote_value < min_quote:
            raise ValueError(f"ERROR [{agent_name}][check_order_valid]: Quote too small on order {order['order_id']} for position {order['position_id']} at {cur_date_int}")

        if abs_base < min_base:
            raise ValueError(f"ERROR [{agent_name}][check_order_valid]: Base too small on order {order['order_id']} for position {order['position_id']} at {cur_date_int}")

        if base_step > 0:
            abs_base_dec = Decimal(str(round(abs_base, 7)))
            base_step_dec = Decimal(str(round(base_step, 7)))
            if abs_base_dec % base_step_dec != 0:
                raise ValueError(f"ERROR [{agent_name}][check_order_valid]: Base is not divisible on order {order['order_id']} for position {order['position_id']} at {cur_date_int}")

    return open_price, side


def check_pending_order_valid(agent_name, has_base_limit, cur_positions, symbol_orders, order, sidx, bid, ask, min_quote, min_base, base_step, cur_date_int):
    # 1. Check base valid for cur_position
    # 2. Check same pos id, same type already have order
    # 3. Check price valid for limit/trigger order
    # 4. Check base valid for symbols_info rule
    pos_id = order['position_id']
    has_position = pos_id in cur_positions.keys()
    base = order['base']
    price = order['price']
    order_type = order['order_type']

    ### Check base valid for cur_position
    # Error 1
    if not has_position:
        side = order['base']>0
        if base==0:
            raise ValueError(f"ERROR [{agent_name}][check_order_valid]: Can't add 0 base on not existed position {pos_id} for order {order['order_id']} at {cur_date_int}")

    # Error 2 & 3
    else:
        pos_sidx = cur_positions[pos_id]['symbol_idx']
        side = cur_positions[pos_id]['base']<0
        if base!=0:
            raise ValueError(f"ERROR [{agent_name}][check_order_valid]: Can't add base on current position {pos_id} for order {order['order_id']} at {cur_date_int}")
        if sidx!=pos_sidx:
            raise ValueError(f"ERROR [{agent_name}][check_order_valid]: Can't close current position {pos_id} by different symbol for order {order['order_id']} at {cur_date_int}")

    ### Check same pos id, same type already have order
    pending_keys = {(porder['position_id'], porder['order_type']) for porder in symbol_orders}
    if (pos_id, order_type) in pending_keys:
        raise ValueError(f"ERROR [{agent_name}][check_order_valid]: Already have pending order with same pos_id {pos_id} and order_type {order_type} for order {order['order_id']} at {cur_date_int}")

    ### Check price valid for limit/trigger order
    if order_type == 1:
        if ((side and price >= ask) or (not side and price <= bid)):
            raise ValueError(f"ERROR [{agent_name}][check_order_valid]: Price not valid on limit order {order['order_id']} for position {order['position_id']} at {cur_date_int}")

    else:
        if ((side and price <= ask) or (not side and price >= bid)):
            print(f"{cur_positions}")
            print(f"price: {price}, ask: {ask}, bid: {bid}, side: {side}")
            raise ValueError(f"ERROR [{agent_name}][check_order_valid]: Price not valid on trigger order {order['order_id']} for position {order['position_id']} at {cur_date_int}")

    ### Check base valid for symbols_info rule
    if has_base_limit and not has_position:
        abs_base = abs(base)
        quote_value = abs_base * price
        if quote_value < min_quote:
            raise ValueError(f"ERROR [{agent_name}][check_order_valid]: Quote too small on order {order['order_id']} for position {order['position_id']} at {cur_date_int}")

        if abs_base < min_base:
            raise ValueError(f"ERROR [{agent_name}][check_order_valid]: Base too small on order {order['order_id']} for position {order['position_id']} at {cur_date_int}")

        if base_step > 0:
            abs_base_dec = Decimal(str(round(abs_base, 7)))
            base_step_dec = Decimal(str(round(base_step, 7)))
            if abs_base_dec % base_step_dec != 0:
                raise ValueError(f"ERROR [{agent_name}][check_order_valid]: Base is not divisible on order {order['order_id']} for position {order['position_id']} at {cur_date_int}")

    return side


cdef bint check_pending_order_touched(int order_type, double price, double bid, double ask, bint side):
    if order_type == 1:
        return ((side and price >= ask) or (not side and price <= bid))

    else:
        return ((side and price <= ask) or (not side and price >= bid))


cdef tuple update_balance(dict cur_positions, double cur_base_balance, double ask, double bid, int cur_sidx, double cur_date_int, list mea_exurs_list, list mea_dates_list):
    cdef:
        double cur_balance = cur_base_balance
        double float_pnl
        double cur_price
        double base

    if cur_positions:
        for pos_id, position_info in cur_positions.items():
            # Update float pnl
            if position_info['symbol_idx']==cur_sidx:
                base = -position_info['base']
                cur_price = ask if base > 0 else bid
                float_pnl = (-base * (cur_price - position_info['open_price'])) - position_info['open_fee']
                cur_positions[pos_id]['float_pnl'] = float_pnl
                cur_positions[pos_id]['mfe'] = float_pnl if float_pnl > cur_positions[pos_id]['mfe'] else cur_positions[pos_id]['mfe']
                cur_positions[pos_id]['mae'] = -float_pnl if -float_pnl > cur_positions[pos_id]['mae'] else cur_positions[pos_id]['mae']

                # MEA
                open_base = position_info['base']
                exur_price = cur_price - position_info['open_price'] if open_base>0 else position_info['open_price'] - cur_price
                mea_exurs_list.append(exur_price)
                mea_dates_list.append(cur_date_int)
            else:
                float_pnl = position_info['float_pnl']

            # Acc cur_balance
            cur_balance += float_pnl

    return cur_positions, cur_balance, mea_exurs_list, mea_dates_list


def c_backtest_mea_core(dict agent_dict, int print_progress=0):
    ### C define
    cdef:
        # Define code temp
        int tidx, cur_sidx, sidx, eqidx, order_sidx
        int total_tick_num
        int progress_unit
        bint has_wfa
        bint has_base_limit
        int cur_wfa_date_idx
        int wfa_dts_len
        np.int64_t[:] wfa_dts_int
        np.int64_t cur_date_int
        int[:] n_symbols_range
        double bid, true_bid
        double ask, true_ask
        np.int64_t[:] date_ints
        double[:] asks
        double[:] bids
        int position_cnt
        int order_type
        double price, price_ask, price_bid, est_spread
        bint side
        bint is_exec, base_not_valid
        bint has_position
        double open_price, slippage

        # Define c trading stuff
        double[:] slippages
        double[:] min_quotes
        double[:] min_bases
        double[:] base_steps
        double cur_base_balance = float(agent_dict['general']['BALANCE'])
        double cur_balance = float(agent_dict['general']['BALANCE'])

    ### Consider base limit
    has_base_limit = 1
    if agent_dict['backtest']['BASE_LIMIT']==0:
        has_base_limit = 0

    ### Symbols info handle
    symbols_info = deepcopy(agent_dict['symbols_info'])

    for sidx, symbol_info in enumerate(symbols_info):
        if has_base_limit==0:
            symbols_info[sidx].update({'MIN_QUOTE': 0, 'MIN_BASE': 0, 'BASE_STEP': 0})
        symbols_info[sidx].update({
        'TRADE_FEE_TYPE': int(symbol_info['TRADE_FEE_TYPE']),
        'TAKER_FEE': float(symbol_info['TAKER_FEE']) if agent_dict['backtest']['NO_COST']!=1 else 0,
        'MAKER_FEE': float(symbol_info['MAKER_FEE']) if agent_dict['backtest']['NO_COST']!=1 else 0,
        'SPREAD_POS': int(symbol_info['SPREAD_POS']),
        'SWAP_TYPE': int(symbol_info['SWAP_TYPE']),
        'LONG_SWAP': float(symbol_info['LONG_SWAP']) if agent_dict['backtest']['NO_COST']!=1 else 0,
        'SHORT_SWAP': float(symbol_info['SHORT_SWAP']) if agent_dict['backtest']['NO_COST']!=1 else 0,
        'SWAP_SETTLE_HOUR': list(map(int, str(symbol_info['SWAP_SETTLE_HOUR']).split(","))),
        'SWAP_SETTLE_DAY': list(map(int, str(symbol_info['SWAP_SETTLE_DAY']).split(",")))
        })

    ### MEA add day tf
    if '1d' not in str(agent_dict['general']['USED_TFS']).split(","):
        agent_dict['general']['USED_TFS'] += ',1d'
        agent_dict['general']['TF_OFFSET'] = str(agent_dict['general']['TF_OFFSET']) + ',0'
        agent_dict['general']['BUFSIZE'] = str(agent_dict['general']['BUFSIZE']) + ',40'

    ### Create agent
    agent_name = agent_dict['general']['AGENT']
    strategy_name = agent_dict['general']['FUNCTION']
    module = import_module('function')
    strategy_class = getattr(module, agent_dict['general']['FUNCTION'])
    agent = strategy_class(agent_name, dummy_log_print)
    agent.load_general_cfg(agent_dict['general'])
    agent.load_strategy_cfg(agent_dict['strategy'])
    agent.load_symbols_info(symbols_info)

    ### Walk forward Analysis initial setting
    has_wfa = 'WFA_PATH' in agent_dict['backtest'].keys()
    if has_wfa:
        # Load wfa datetimes
        wfa_cfgs_path = agent_dict['backtest']['WFA_PATH'] + '/'
        date_files = glob.glob(f"{wfa_cfgs_path}"+'[0-9]*')
        wfa_dates = [os.path.splitext(os.path.basename(file))[0] for file in date_files]
        if not wfa_dates:
            raise ValueError(f"ERROR [{agent_name}]: Backtest Error: No wfa config is read")
        wfa_dts = [np.datetime64(datetime.strptime(date, '%Y-%m-%d-%H-%M-%S')) for date in wfa_dates]
        wfa_dts = list(np.sort(wfa_dts))
        wfa_dts_len = len(wfa_dts)

        # if only use wfa then change START_DATE
        if wfa_dts[0] > agent_dict['backtest']['START_DATE'] and agent_dict['backtest']['USED_ONLY_WFA']==1:
            agent_dict['backtest']['START_DATE'] = wfa_dts[0]

        # Get the first wfa config if START_DATE over first wfa datetime
        cur_wfa_date_idx = get_leftmost_time_idx(agent_dict['backtest']['START_DATE'], wfa_dts)
        if cur_wfa_date_idx!=-1:
            # Get all cfgs
            wfa_cfgs = pd.read_excel(wfa_cfgs_path + wfa_dates[cur_wfa_date_idx] + '.xlsx', sheet_name=None)

            # Load strategy config if it exists
            wfa_strategy_cfg_names = [x for x in list(wfa_cfgs.keys()) if x.startswith('strategy')]
            if wfa_strategy_cfg_names:
                wfa_strategy_cfg = wfa_cfgs[wfa_strategy_cfg_names[0]].to_dict('records')[0]
                agent.load_strategy_cfg(wfa_strategy_cfg)

            # Load general if general config exists
            if 'general' in list(wfa_cfgs.keys()):
                wfa_general_cfg = wfa_cfgs['general'].to_dict('records')[0]
                agent.load_general_cfg( wfa_general_cfg)

        # Change wfa dts to cython version
        wfa_dts_int = np.array(wfa_dts).astype('datetime64[ns]').astype('int64')

    ### Initialize backtest stuff
    default_order = dict(position_cnt = 0, symbol_idx = 0, position_id = 0, order_id = 0, order_type = 0, base = 0, price = 0,
                         oco_orders = [], oco_cancel_ids = [], tag = '', invalid = 'NO', valid = 0, cancel = 0)
    symbols = str(agent_dict['general']['SYMBOLS']).split(",")
    n_symbols_range = np.arange(len(symbols), dtype=np.int32)
    orders_info_list = []
    pending_orders = [[] for _ in n_symbols_range]
    cur_positions = dict()  # dict of dict
    position_cnt = -1
    slippages = np.array([float(symbols_info[sidx]['SLIPPAGE']) for sidx in n_symbols_range], dtype=np.float64)
    min_quotes = np.array([float(symbols_info[sidx]['MIN_QUOTE']) for sidx in n_symbols_range], dtype=np.float64)
    min_bases = np.array([float(symbols_info[sidx]['MIN_BASE']) for sidx in n_symbols_range], dtype=np.float64)
    base_steps = np.array([float(symbols_info[sidx]['BASE_STEP']) for sidx in n_symbols_range], dtype=np.float64)
    date_ints = np.zeros(len(n_symbols_range), dtype=np.int64)
    asks = np.zeros(len(n_symbols_range), dtype=np.float64)
    bids = np.zeros(len(n_symbols_range), dtype=np.float64)

    ### MEA Initialize
    mea_dates_list = []
    mea_exurs_list = []

    ### Load df_set
    if 'df_sets' in agent_dict.keys():
        df_sets = agent_dict['df_sets']
    else:
        df_sets = read_df_sets(agent_dict)

    ### Initialize tick gen
    tick_gen = TickGen(agent_dict)
    total_tick_num = tick_gen.total_tick_num
    progress_unit = total_tick_num//10

    ### MEA calc 2 types of atr
    used_tfs = str(agent_dict['general']['USED_TFS']).split(',')
    if 'MAIN_TF' not in agent_dict['strategy'].keys(): # use min tf for atr tf
        atr_tf_idx = 0
    else:
        atr_tf_idx = used_tfs.index(agent_dict['strategy']['MAIN_TF'])

    if 'MEA_ATR_LEN' not in agent_dict['backtest'].keys():
        atr_len = 25
        datr_len = 14
    else:
        atr_len = int(agent_dict['backtest']['MEA_ATR_LEN'])
        datr_len = int(agent_dict['backtest']['MEA_ATR_LEN'])

    atr_df = df_sets[0][atr_tf_idx]
    atr_df = AverageTrueRange(high=atr_df['High'],
                              low=atr_df['Low'],
                              close=atr_df['Close'],
                              window=atr_len).average_true_range()

    datr_tf_idx = used_tfs.index('1d')
    datr_df = df_sets[0][datr_tf_idx]
    datr_df = AverageTrueRange(high=datr_df['High'],
                              low=datr_df['Low'],
                              close=datr_df['Close'],
                              window=datr_len).average_true_range()

    ### Initialize index tracker
    kidx_tracker = kidxTracker(df_sets)

    ### Prepare agent
    agent.load_df_sets(df_sets)
    agent.indicator_calc()

    ### Initialize EQTracker
    eq_tf = str(agent_dict['general']['USED_TFS']).split(",")[0]
    eq_tracker = EQTracker(eq_tf)

    ### For loop ticks
    for tidx in range(total_tick_num):
        # Print progress bar
        if print_progress==1:
            if tidx%progress_unit==0 or tidx==(total_tick_num-1):
                printProgressBar(tidx, total_tick_num-1, prefix = 'Backtest:', suffix = 'Complete', length = 50)

        # Get new tick and update k index
        cur_sidx, cur_date_int, bid, ask = tick_gen.get_tick()
        kidx_tracker.update_kidx(cur_sidx, cur_date_int)
        kidx_sets = kidx_tracker.kidx_sets_out # MEA

        # Update latest_date, latest_ask, latest_bid
        date_ints[cur_sidx] = cur_date_int
        asks[cur_sidx] = ask
        bids[cur_sidx] = bid

        # Check pending_orders for current tick symbol
        symbol_orders = pending_orders[cur_sidx]
        if symbol_orders:
            remove_oidx = []
            for oidx, order in enumerate(symbol_orders):
                if order['valid'] == 1:
                    # For valid==0 order, don't need to consider in remove_oidx. Because in live, it is in new_orders buffer. OCO only check pending orders.
                    if oidx not in remove_oidx:
                        # Check if limit/trigger order is executed at current tick
                        side = order['side']
                        order_type = order['order_type']
                        price = order['price']
                        is_exec = check_pending_order_touched(order_type, price, bid, ask, side)

                        if is_exec:
                            remove_oidx.append(oidx)

                            # Calculate slippage
                            if order_type==2:
                                slippage = slippages[cur_sidx] if side else -slippages[cur_sidx]
                                open_price = price * (1 + slippage)
                            else:
                                slippage = 0
                                open_price = price

                            # Update info
                            order.update({'exec_price': open_price, 'exec_date': cur_date_int})

                            # Update position
                            order, cur_positions, position_cnt, cur_base_balance, mea_exurs_list, mea_dates_list = cur_positions_change(agent_name, order, cur_positions, position_cnt, symbols_info, cur_base_balance, ask, bid, slippage, cur_date_int, mea_exurs_list, mea_dates_list, kidx_sets, atr_df, datr_df, atr_tf_idx, datr_tf_idx)

                            # oco cancel
                            if order['oco_cancel_ids']:
                                order_oco_cancel_ids = order['oco_cancel_ids']

                                # Remove from pending_order for this symbol and add to orders_info_list
                                if any(porder['order_id'] in order_oco_cancel_ids for porder in symbol_orders):  # Fix zip error
                                    cancel_orders, remove_pidx = zip(*[({**porder, 'exec_date': cur_date_int, 'cancel': 1}, pidx) \
                                    for pidx, porder in enumerate(symbol_orders) if porder['order_id'] in order_oco_cancel_ids])
                                    orders_info_list.extend(cancel_orders)
                                    remove_oidx.extend(remove_pidx)

                                # Remove from other symbol pending_order and add to orders_info_list
                                for sidx in n_symbols_range:
                                    if sidx!=cur_sidx:
                                        ssymbol_orders = pending_orders[sidx]
                                        cancel_orders = [{**porder, 'exec_date': cur_date_int, 'cancel': 1} \
                                                          for porder in ssymbol_orders if porder['order_id'] in order_oco_cancel_ids]
                                        orders_info_list.extend(cancel_orders)
                                        pending_orders[sidx] = [porder for porder in ssymbol_orders if porder['order_id'] not in order_oco_cancel_ids]

                            # oco add, need add oco_trigger price for those orders
                            if order['oco_orders']:
                                est_spread = ask - bid           # The spread here is estimated, not exact the actual spread.
                                price_ask = open_price if side else price + est_spread
                                price_bid = open_price - est_spread if side else price
                                add_orders = order['oco_orders']
                                add_orders = [{**aorder, 'oco_price_bid': price_bid, 'oco_price_ask': price_ask} for aorder in add_orders]
                                new_orders.extend(add_orders)

                            # Finally, append to order_info_list
                            orders_info_list.append(order)

                # For valid=0 order, add to new_orders
                else:
                    remove_oidx.append(oidx)
                    new_orders.append(order)

            # Delete orders from pending_orders for this symbols if executed or cancelled or valid=0
            if remove_oidx:
                pending_orders[cur_sidx] = [porder for pidx, porder in enumerate(symbol_orders) if pidx not in remove_oidx]

            # New orders processing, reprocessing when it gets oco add orders or valid=0 orders
            new_orders = [{**default_order, **norder} for norder in new_orders]
            new_orders.sort(key=lambda x: x['order_type']) # Sort to prioritize market orders first
            while new_orders:
                order = new_orders.pop(0)

                if order['symbol_idx']==cur_sidx:
                    # For oco added orders, the market detected price is not newest price but the triggered price
                    if 'oco_price_bid' in order.keys():
                        true_bid = order['oco_price_bid']
                        true_ask = order['oco_price_ask']
                    else:
                        true_bid = bid
                        true_ask = ask

                    # Process market
                    if order['order_type']==0:
                        # Check market order valid
                        open_price, side = check_market_order_valid(agent_name, has_base_limit, cur_positions, order, cur_sidx, true_bid, true_ask, min_quotes[cur_sidx], min_bases[cur_sidx], base_steps[cur_sidx], cur_date_int)

                        # Calculate slippage
                        slippage = slippages[cur_sidx] if side else -slippages[cur_sidx]
                        open_price = open_price * (1 + slippage)

                        # Update info
                        order.update({'pending_date': cur_date_int, 'exec_price': open_price, 'exec_date': cur_date_int})

                        # Update position
                        order, cur_positions, position_cnt, cur_base_balance, mea_exurs_list, mea_dates_list = cur_positions_change(agent_name, order, cur_positions, position_cnt, symbols_info, cur_base_balance, true_ask, true_bid, slippage, cur_date_int, mea_exurs_list, mea_dates_list, kidx_sets, atr_df, datr_df, atr_tf_idx, datr_tf_idx)

                        # oco cancel
                        if order['oco_cancel_ids']:
                            order_oco_cancel_ids = order['oco_cancel_ids']

                            # Remove from pending_order and add to orders_info_list
                            for sidx in n_symbols_range:
                                symbol_orders = pending_orders[sidx]
                                cancel_orders = [{**porder, 'exec_date': cur_date_int, 'cancel': 1} \
                                                  for porder in symbol_orders if porder['order_id'] in order_oco_cancel_ids]
                                orders_info_list.extend(cancel_orders)
                                pending_orders[sidx] = [porder for porder in symbol_orders if porder['order_id'] not in order_oco_cancel_ids]

                        # oco add
                        if order['oco_orders']:
                            add_orders = order['oco_orders']
                            add_orders = [{**default_order, **aorder} for aorder in add_orders]
                            new_orders.extend(add_orders)

                        # Finally, append to order_info_list
                        orders_info_list.append(order)

                    # Process limit & trigger
                    else:
                        # Check pending order valid
                        side = check_pending_order_valid(agent_name, has_base_limit, cur_positions, pending_orders[cur_sidx], order, cur_sidx, true_bid, true_ask, min_quotes[cur_sidx], min_bases[cur_sidx], base_steps[cur_sidx], cur_date_int)

                        # Update info
                        order.update({'pending_date': cur_date_int, 'valid': 1, 'side': side})

                        # For oco added orders, need recheck pending trigger
                        if 'oco_price_bid' in order.keys():
                            # Check if limit/trigger order is executed at current tick
                            order_type = order['order_type']
                            price = order['price']
                            is_exec = check_pending_order_touched(order_type, price, bid, ask, side)

                            if is_exec:
                                # Calculate slippage
                                if order_type==2:
                                    slippage = slippages[cur_sidx] if side else -slippages[cur_sidx]
                                    open_price = price * (1 + slippage)
                                else:
                                    slippage = 0
                                    open_price = price

                                # Update info
                                order.update({'exec_price': open_price, 'exec_date': cur_date_int})

                                # Update position
                                order, cur_positions, position_cnt, cur_base_balance, mea_exurs_list, mea_dates_list = cur_positions_change(agent_name, order, cur_positions, position_cnt, symbols_info, cur_base_balance, ask, bid, slippage, cur_date_int, mea_exurs_list, mea_dates_list, kidx_sets, atr_df, datr_df, atr_tf_idx, datr_tf_idx)

                                # oco cancel
                                if order['oco_cancel_ids']:
                                    order_oco_cancel_ids = order['oco_cancel_ids']

                                    # Remove from pending_order and add to orders_info_list
                                    for sidx in n_symbols_range:
                                        symbol_orders = pending_orders[sidx]
                                        cancel_orders = [{**porder, 'exec_date': cur_date_int, 'cancel': 1} \
                                                          for porder in symbol_orders if porder['order_id'] in order_oco_cancel_ids]
                                        orders_info_list.extend(cancel_orders)
                                        pending_orders[sidx] = [porder for porder in symbol_orders if porder['order_id'] not in order_oco_cancel_ids]

                                # oco add
                                if order['oco_orders']:
                                    add_orders = order['oco_orders']
                                    add_orders = [{**default_order, **aorder} for aorder in add_orders]
                                    new_orders.extend(add_orders)

                                # Finally, append to order_info_list
                                orders_info_list.append(order)
                            else:
                                # Append to pending_orders
                                pending_orders[cur_sidx].append(order)

                # For different symbol order, if previous date = cur_date_int, than open or pend with valid=1
                # Else add to pending_orders with valid=0
                else:
                    order_sidx = order['symbol_idx']
                    if cur_date_int==date_ints[order_sidx]:
                        # Get info
                        true_bid = bids[order_sidx]
                        true_ask = asks[order_sidx]

                        # Process market
                        if order['order_type']==0:
                            # Check market order valid
                            open_price, side = check_market_order_valid(agent_name, has_base_limit, cur_positions, order, order_sidx, true_bid, true_ask, min_quotes[order_sidx], min_bases[order_sidx], base_steps[order_sidx], cur_date_int)

                            # Calculate slippage
                            slippage = slippages[order_sidx] if side else -slippages[order_sidx]
                            open_price = open_price * (1 + slippage)

                            # Update info
                            order.update({'pending_date': cur_date_int, 'exec_price': open_price, 'exec_date': cur_date_int})

                            # Update position
                            order, cur_positions, position_cnt, cur_base_balance, mea_exurs_list, mea_dates_list = cur_positions_change(agent_name, order, cur_positions, position_cnt, symbols_info, cur_base_balance, true_ask, true_bid, slippage, cur_date_int, mea_exurs_list, mea_dates_list, kidx_sets, atr_df, datr_df, atr_tf_idx, datr_tf_idx)

                            # oco cancel
                            if order['oco_cancel_ids']:
                                order_oco_cancel_ids = order['oco_cancel_ids']

                                # Remove from pending_order and add to orders_info_list
                                for sidx in n_symbols_range:
                                    symbol_orders = pending_orders[sidx]
                                    cancel_orders = [{**porder, 'exec_date': cur_date_int, 'cancel': 1} \
                                                      for porder in symbol_orders if porder['order_id'] in order_oco_cancel_ids]
                                    orders_info_list.extend(cancel_orders)
                                    pending_orders[sidx] = [porder for porder in symbol_orders if porder['order_id'] not in order_oco_cancel_ids]

                            # oco add
                            if order['oco_orders']:
                                add_orders = order['oco_orders']
                                add_orders = [{**default_order, **aorder} for aorder in add_orders]
                                new_orders.extend(add_orders)

                            # Finally, append to order_info_list
                            orders_info_list.append(order)

                        # Process limit & trigger
                        else:
                            # Check pending order valid
                            side = check_pending_order_valid(agent_name, has_base_limit, cur_positions, pending_orders[order_sidx], order, order_sidx, true_bid, true_ask, min_quotes[order_sidx], min_bases[order_sidx], base_steps[order_sidx], cur_date_int)

                            # Update info
                            order.update({'pending_date': cur_date_int, 'valid': 1, 'side': side})

                            # append to pending_orders
                            pending_orders[order_sidx].append(order)
                    else:
                        order['pending_date'] = cur_date_int
                        pending_orders[order_sidx].append(order)

        # wfa Update paras
        if has_wfa:
            if cur_wfa_date_idx < wfa_dts_len-1:
                if cur_date_int > wfa_dts_int[cur_wfa_date_idx+1]:
                    cur_wfa_date_idx+=1

                    # Get all cfgs
                    wfa_cfgs = pd.read_excel(wfa_cfgs_path + wfa_dates[cur_wfa_date_idx] + '.xlsx', sheet_name=None)

                    # Load strategy config if it exists
                    wfa_strategy_cfg_names = [x for x in list(wfa_cfgs.keys()) if x.startswith('strategy')]
                    if wfa_strategy_cfg_names:
                        wfa_strategy_cfg = wfa_cfgs[wfa_strategy_cfg_names[0]].to_dict('records')[0]
                        agent.load_strategy_cfg(wfa_strategy_cfg)
                        agent.indicator_calc()

                    # Load general if general config exists
                    if 'general' in list(wfa_cfgs.keys()):
                        wfa_general_cfg = wfa_cfgs['general'].to_dict('records')[0]
                        agent.load_general_cfg( wfa_general_cfg)

        # Update cur_balance
        cur_positions, cur_balance, mea_exurs_list, mea_dates_list = update_balance(cur_positions, cur_base_balance, ask, bid, cur_sidx, cur_date_int, mea_exurs_list, mea_dates_list)
        eq_tracker.update_balance(cur_balance, cur_date_int)

        # agent on_tick action
        new_orders, cancel_order_ids = agent.on_tick(cur_sidx, cur_date_int, bid, ask, kidx_tracker.kidx_sets_out, kidx_tracker.kidx_changed_flags, pending_orders, cur_positions, cur_balance)

        # Cancel order from pending_orders
        for sidx in n_symbols_range:
            symbol_orders = pending_orders[sidx]
            cancel_orders = [{**porder, 'exec_date': cur_date_int, 'cancel': 1} \
                              for porder in symbol_orders if porder['order_id'] in cancel_order_ids]
            orders_info_list.extend(cancel_orders)
            pending_orders[sidx] = [porder for porder in symbol_orders if porder['order_id'] not in cancel_order_ids]

        # New orders processing, reprocessing when it gets oco add orders
        new_orders = [{**default_order, **norder} for norder in new_orders]
        new_orders.sort(key=lambda x: x['order_type'])  # Sort to prioritize market orders first
        while new_orders:
            order = new_orders.pop(0)

            if order['symbol_idx']==cur_sidx:
                # Process market
                if order['order_type']==0:
                    # Check market order valid
                    open_price, side = check_market_order_valid(agent_name, has_base_limit, cur_positions, order, cur_sidx, bid, ask, min_quotes[cur_sidx], min_bases[cur_sidx], base_steps[cur_sidx], cur_date_int)

                    # Calculate slippage
                    slippage = slippages[cur_sidx] if side else -slippages[cur_sidx]
                    open_price = open_price * (1 + slippage)

                    # Update info
                    order.update({'pending_date': cur_date_int, 'exec_price': open_price, 'exec_date': cur_date_int})

                    # Update position
                    order, cur_positions, position_cnt, cur_base_balance, mea_exurs_list, mea_dates_list = cur_positions_change(agent_name, order, cur_positions, position_cnt, symbols_info, cur_base_balance, ask, bid, slippage, cur_date_int, mea_exurs_list, mea_dates_list, kidx_sets, atr_df, datr_df, atr_tf_idx, datr_tf_idx)

                    # oco cancel
                    if order['oco_cancel_ids']:
                        order_oco_cancel_ids = order['oco_cancel_ids']

                        # Remove from pending_order and add to orders_info_list
                        for sidx in n_symbols_range:
                            symbol_orders = pending_orders[sidx]
                            cancel_orders = [{**porder, 'exec_date': cur_date_int, 'cancel': 1} \
                                              for porder in symbol_orders if porder['order_id'] in order_oco_cancel_ids]
                            orders_info_list.extend(cancel_orders)
                            pending_orders[sidx] = [porder for porder in symbol_orders if porder['order_id'] not in order_oco_cancel_ids]

                    # oco add
                    if order['oco_orders']:
                        add_orders = order['oco_orders']
                        add_orders = [{**default_order, **aorder} for aorder in add_orders]
                        new_orders.extend(add_orders)

                    # Finally, append to order_info_list
                    orders_info_list.append(order)

                # Process limit & trigger
                else:
                    # Check pending order valid
                    side = check_pending_order_valid(agent_name, has_base_limit, cur_positions, pending_orders[cur_sidx], order, cur_sidx, bid, ask, min_quotes[cur_sidx], min_bases[cur_sidx], base_steps[cur_sidx], cur_date_int)

                    # Update info
                    order.update({'pending_date': cur_date_int, 'valid': 1, 'side': side})

                    # append to pending_orders
                    pending_orders[cur_sidx].append(order)

            # For different symbol order, if previous date = cur_date_int, than open or pend with valid=1
            # Else add to pending_orders with valid=0
            else:
                order_sidx = order['symbol_idx']
                if cur_date_int==date_ints[order_sidx]:
                    # Get info
                    true_bid = bids[order_sidx]
                    true_ask = asks[order_sidx]

                    # Process market
                    if order['order_type']==0:
                        # Check market order valid
                        open_price, side = check_market_order_valid(agent_name, has_base_limit, cur_positions, order, order_sidx, true_bid, true_ask, min_quotes[order_sidx], min_bases[order_sidx], base_steps[order_sidx], cur_date_int)

                        # Calculate slippage
                        slippage = slippages[order_sidx] if side else -slippages[order_sidx]
                        open_price = open_price * (1 + slippage)

                        # Update info
                        order.update({'pending_date': cur_date_int, 'exec_price': open_price, 'exec_date': cur_date_int})

                        # Update position
                        order, cur_positions, position_cnt, cur_base_balance, mea_exurs_list, mea_dates_list = cur_positions_change(agent_name, order, cur_positions, position_cnt, symbols_info, cur_base_balance, true_ask, true_bid, slippage, cur_date_int, mea_exurs_list, mea_dates_list, kidx_sets, atr_df, datr_df, atr_tf_idx, datr_tf_idx)

                        # oco cancel
                        if order['oco_cancel_ids']:
                            order_oco_cancel_ids = order['oco_cancel_ids']

                            for sidx in n_symbols_range:
                                symbol_orders = pending_orders[sidx]
                                cancel_orders = [{**porder, 'exec_date': cur_date_int, 'cancel': 1} \
                                                  for porder in symbol_orders if porder['order_id'] in order_oco_cancel_ids]
                                orders_info_list.extend(cancel_orders)
                                pending_orders[sidx] = [porder for porder in symbol_orders if porder['order_id'] not in order_oco_cancel_ids]

                        # oco add
                        if order['oco_orders']:
                            add_orders = order['oco_orders']
                            add_orders = [{**default_order, **aorder} for aorder in add_orders]
                            new_orders.extend(add_orders)
    
                        # Finally, append to order_info_list
                        orders_info_list.append(order)

                    # Process limit & trigger
                    else:
                        # Check pending order valid
                        side = check_pending_order_valid(agent_name, has_base_limit, cur_positions, pending_orders[order_sidx], order, order_sidx, true_bid, true_ask, min_quotes[order_sidx], min_bases[order_sidx], base_steps[order_sidx], cur_date_int)

                        # Update info
                        order.update({'pending_date': cur_date_int, 'valid': 1, 'side': side})

                        # append to pending_orders
                        pending_orders[order_sidx].append(order)
                else:
                    # For other symbol that cur_date is old, add to pending_orders with valid=0 and no pending_date
                    pending_orders[order_sidx].append(order)

    ### handle orders_info
    orders_info = pd.DataFrame(orders_info_list,
                               columns=['position_cnt', 'symbol_idx', 'position_id', 'order_id', 'order_type', 'base', 'price', 'tag',
                                        'invalid', 'cancel', 'pending_date', 'exec_price', 'exec_date', 'exec_base', 'trade_fee',
                                        'spread_fee', 'slippage_fee', 'swap_fee', 'pnl', 'mfe', 'mae',
                                        'mea_mfe_price', 'mea_bmfe_price', 'mea_cmfe_price', 'mea_mae_price', 'mea_pnl_price', 'mea_mhl_price',  # MEA
                                        'mea_mfe_per', 'mea_bmfe_per', 'mea_cmfe_per', 'mea_mae_per', 'mea_pnl_per', 'mea_mhl_per',  # MEA
                                        'mea_mfe_atr', 'mea_bmfe_atr', 'mea_cmfe_atr', 'mea_mae_atr', 'mea_pnl_atr', 'mea_mhl_atr',  # MEA
                                        'mea_mfe_datr', 'mea_bmfe_datr', 'mea_cmfe_datr', 'mea_mae_datr', 'mea_pnl_datr', 'mea_mhl_datr', # MEA
                                        'mea_mfe_date', 'mea_bmfe_date', 'mea_cmfe_date', 'mea_mae_date'] # MEA
                               )
    if orders_info_list:
        orders_info['pending_date'] = pd.to_datetime(orders_info['pending_date'])
        orders_info['exec_date'] = pd.to_datetime(orders_info['exec_date'])
        orders_info['mea_mfe_date'] = pd.to_datetime(orders_info['mea_mfe_date']) # MEA
        orders_info['mea_bmfe_date'] = pd.to_datetime(orders_info['mea_bmfe_date']) # MEA
        orders_info['mea_cmfe_date'] = pd.to_datetime(orders_info['mea_cmfe_date']) # MEA
        orders_info['mea_mae_date'] = pd.to_datetime(orders_info['mea_mae_date']) # MEA
        orders_info = orders_info.fillna(0)

    ### handle eq_series
    eq_series = eq_tracker.get_eq()

    ### Return
    agent_dict.update({'orders_info': orders_info, 'eq_series': eq_series})
    return agent_dict
