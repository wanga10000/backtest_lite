class function_base:
    def load_general_cfg(self, general_cfg):
        pass


    def load_strategy_cfg(self, strategy_cfg):
        pass


    def load_symbols_info(self, symbols_info):
        pass


    def load_df_sets(self, df_sets):
        pass


    def declare_df(self):
        pass


    def indicator_calc(self):
        pass


    def on_tick(self, cur_sidx, cur_date_int, bid, ask, kidx_sets, kidx_changed_flags, pending_orders, cur_positions, cur_balance):
        new_orders = []
        cancel_order_ids = []

        return new_orders, cancel_order_ids