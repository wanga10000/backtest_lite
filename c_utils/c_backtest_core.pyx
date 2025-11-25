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
from decimal import Decimal


def dummy_log_print(msg, to_tg=0):
    # Dummy log_print function that does nothing
    pass


def cur_positions_change(agent_name, order, cur_positions, position_cnt, symbols_info, cur_base_balance, ask, bid, slippage, cur_date_int):
    if order['position_id'] in cur_positions.keys():
        # Check base is 0
        if order['base'] != 0:
            raise ValueError(f"ERROR [{agent_name}][cur_positions_change]: Can't add base on current position {order['position_id']} at {cur_date_int}")

        position_info = cur_positions.pop(order['position_id'])

        # Check sidx the same
        sidx = order['symbol_idx']
        if sidx != position_info['symbol_idx']:
            raise ValueError(f"ERROR [{agent_name}][cur_positions_change]: Can't close current position {order['position_id']} by different symbol at {cur_date_int}")

        # Close fee calculate: 'trade_fee', 'spread_fee', 'slippage_fee'
        symbol_info = symbols_info[sidx]
        TRADE_FEE_TYPE = symbol_info['TRADE_FEE_TYPE']
        base = -position_info['base']
        trade_fee_rate = symbol_info['TAKER_FEE']
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

        cur_base_balance += pnl
        order.update({'position_cnt': position_info['position_cnt'], 'exec_base': base, 'trade_fee': trade_fee,
                  'spread_fee': spread_fee, 'slippage_fee': slippage_fee, 'swap_fee': swap_fee, 'pnl': pnl, 'mfe': mfe, 'mae': mae})
    else:
        # Check base is not 0
        if order['base'] == 0:
            raise ValueError(f"ERROR [{agent_name}][cur_positions_change]: Can't add 0 base on not existed position {order['position_id']} for order {order['order_id']} at {cur_date_int}")

        # Acc postion cnt
        position_cnt += 1

        # Open fee calculate: 'trade_fee', 'spread_fee', 'slippage_fee'
        symbol_info = symbols_info[order['symbol_idx']]
        TRADE_FEE_TYPE = symbol_info['TRADE_FEE_TYPE']
        base = order['base']
        trade_fee_rate = symbol_info['TAKER_FEE']
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
                                                   base=base, open_fee=trade_fee, float_pnl=-trade_fee, mfe=0, mae=0)
        order.update({'position_cnt': position_cnt, 'exec_base': base, 'trade_fee': trade_fee,
                      'spread_fee': spread_fee, 'slippage_fee': slippage_fee, 'swap_fee': 0})

    return order, cur_positions, position_cnt, cur_base_balance


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
            raise ValueError(f"ERROR [{agent_name}][check_order_valid]: Can't add 0 base on not existed position {pos_id} at {cur_date_int}")

    # Error 2 & 3
    else:
        pos_sidx = cur_positions[pos_id]['symbol_idx']
        side = cur_positions[pos_id]['base']<0
        if base!=0:
            raise ValueError(f"ERROR [{agent_name}][check_order_valid]: Can't add base on current position {pos_id} at {cur_date_int}")
        if sidx!=pos_sidx:
            raise ValueError(f"ERROR [{agent_name}][check_order_valid]: Can't close current position {pos_id} by different symbol at {cur_date_int}")

    ### Check base valid for symbols_info rule
    open_price = ask if side else bid
    if has_base_limit and not has_position:
        abs_base = abs(base)
        quote_value = abs_base * open_price
        if quote_value < min_quote:
            raise ValueError(f"ERROR [{agent_name}][check_order_valid]: Quote too small for position {order['position_id']} at {cur_date_int}")

        if abs_base < min_base:
            raise ValueError(f"ERROR [{agent_name}][check_order_valid]: Base too small for position {order['position_id']} at {cur_date_int}")

        if base_step > 0:
            abs_base_dec = Decimal(str(round(abs_base, 7)))
            base_step_dec = Decimal(str(round(base_step, 7)))
            if abs_base_dec % base_step_dec != 0:
                raise ValueError(f"ERROR [{agent_name}][check_order_valid]: Base is not divisible for position {order['position_id']} at {cur_date_int}")

    return open_price, side





cdef tuple update_balance(dict cur_positions, double cur_base_balance, double ask, double bid, int cur_sidx):
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

            else:
                float_pnl = position_info['float_pnl']

            # Acc cur_balance
            cur_balance += float_pnl

    return cur_positions, cur_balance


def c_backtest_core(dict agent_dict, int print_progress=0):
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
        double bid
        double ask
        int position_cnt
        bint side
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
    default_order = dict(position_cnt = 0, symbol_idx = 0, position_id = 0, order_id = 0, base = 0, tag = '', invalid = 'NO', cancel = 0)
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

    ### Load df_set
    if 'df_sets' in agent_dict.keys():
        df_sets = agent_dict['df_sets']
    else:
        df_sets = read_df_sets(agent_dict)

    ### Initialize tick gen
    tick_gen = TickGen(agent_dict)
    total_tick_num = tick_gen.total_tick_num
    progress_unit = total_tick_num//10

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

        # Update latest_date, latest_ask, latest_bid
        date_ints[cur_sidx] = cur_date_int
        asks[cur_sidx] = ask
        bids[cur_sidx] = bid

        # Process pending orders for current symbol
        if pending_orders[cur_sidx]:
            for order in pending_orders[cur_sidx]:
                # Check market order valid
                open_price, side = check_market_order_valid(agent_name, has_base_limit, cur_positions, order, cur_sidx, bid, ask, min_quotes[cur_sidx], min_bases[cur_sidx], base_steps[cur_sidx], cur_date_int)

                # Calculate slippage
                slippage = slippages[cur_sidx] if side else -slippages[cur_sidx]
                open_price = open_price * (1 + slippage)

                # Update info
                order.update({'exec_price': open_price, 'exec_date': cur_date_int})

                # Update position
                order, cur_positions, position_cnt, cur_base_balance = cur_positions_change(agent_name, order, cur_positions, position_cnt, symbols_info, cur_base_balance, ask, bid, slippage, cur_date_int)

                # Append to order_info_list
                orders_info_list.append(order)
            
            # Clear pending orders for this symbol
            pending_orders[cur_sidx] = []

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
        cur_positions, cur_balance = update_balance(cur_positions, cur_base_balance, ask, bid, cur_sidx)
        eq_tracker.update_balance(cur_balance, cur_date_int)

        # agent on_tick action
        new_orders, cancel_order_ids = agent.on_tick(cur_sidx, cur_date_int, bid, ask, kidx_tracker.kidx_sets_out, kidx_tracker.kidx_changed_flags, pending_orders, cur_positions, cur_balance)

        # Cancel orders from pending_orders
        for sidx in n_symbols_range:
            symbol_orders = pending_orders[sidx]
            cancel_orders = [porder for porder in symbol_orders if porder['order_id'] in cancel_order_ids]
            for cancel_order in cancel_orders:
                cancel_order['cancel'] = 1
                cancel_order['exec_date'] = cur_date_int
                orders_info_list.append(cancel_order)
            pending_orders[sidx] = [porder for porder in symbol_orders if porder['order_id'] not in cancel_order_ids]

        # Process new orders (only market orders now)
        new_orders = [{**default_order, **norder} for norder in new_orders]
        for order in new_orders:
            order_sidx = order['symbol_idx']
            
            # If order is for current symbol, execute immediately
            if order_sidx == cur_sidx:
                # Check market order valid
                open_price, side = check_market_order_valid(agent_name, has_base_limit, cur_positions, order, order_sidx, bid, ask, min_quotes[order_sidx], min_bases[order_sidx], base_steps[order_sidx], cur_date_int)

                # Calculate slippage
                slippage = slippages[order_sidx] if side else -slippages[order_sidx]
                open_price = open_price * (1 + slippage)

                # Update info
                order.update({'pending_date': cur_date_int, 'exec_price': open_price, 'exec_date': cur_date_int})

                # Update position
                order, cur_positions, position_cnt, cur_base_balance = cur_positions_change(agent_name, order, cur_positions, position_cnt, symbols_info, cur_base_balance, ask, bid, slippage, cur_date_int)

                # Append to order_info_list
                orders_info_list.append(order)
            
            # If order is for different symbol
            else:
                # If that symbol has appeared in this tick cycle, execute with its last price
                if cur_date_int == date_ints[order_sidx]:
                    true_bid = bids[order_sidx]
                    true_ask = asks[order_sidx]
                    
                    # Check market order valid
                    open_price, side = check_market_order_valid(agent_name, has_base_limit, cur_positions, order, order_sidx, true_bid, true_ask, min_quotes[order_sidx], min_bases[order_sidx], base_steps[order_sidx], cur_date_int)

                    # Calculate slippage
                    slippage = slippages[order_sidx] if side else -slippages[order_sidx]
                    open_price = open_price * (1 + slippage)

                    # Update info
                    order.update({'pending_date': cur_date_int, 'exec_price': open_price, 'exec_date': cur_date_int})

                    # Update position
                    order, cur_positions, position_cnt, cur_base_balance = cur_positions_change(agent_name, order, cur_positions, position_cnt, symbols_info, cur_base_balance, true_ask, true_bid, slippage, cur_date_int)

                    # Append to order_info_list
                    orders_info_list.append(order)
                
                # Otherwise, add to pending orders to execute when that symbol's tick arrives
                else:
                    order['pending_date'] = cur_date_int
                    pending_orders[order_sidx].append(order)

    ### handle orders_info
    orders_info = pd.DataFrame(orders_info_list,
                               columns=['position_cnt', 'symbol_idx', 'position_id', 'order_id', 'base', 'tag', 'invalid', 'cancel',
                                        'pending_date', 'exec_price', 'exec_date', 'exec_base', 'trade_fee',
                                        'spread_fee', 'slippage_fee', 'swap_fee', 'pnl', 'mfe', 'mae']
                               )
    if orders_info_list:
        orders_info['pending_date'] = pd.to_datetime(orders_info['pending_date'])
        orders_info['exec_date'] = pd.to_datetime(orders_info['exec_date'])
        orders_info = orders_info.fillna(0)

    ### handle eq_series
    eq_series = eq_tracker.get_eq()

    ### Return
    agent_dict.update({'orders_info': orders_info, 'eq_series': eq_series})
    return agent_dict
