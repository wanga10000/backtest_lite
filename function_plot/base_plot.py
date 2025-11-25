from utils.utils_general import read_df_sets
from lightweight_charts import Chart


def base_plot(agent_dicts_dict):
    global lines, agent_name_sel, orders_info, df_set, USED_TFS

    # Get all possible USED_TFS
    used_tfs_all = []
    for agent_dict in agent_dicts_dict.values():
        for tf in str(agent_dict['general']['USED_TFS']).split(","):
            if tf not in used_tfs_all:
                used_tfs_all.append(tf)

    # Load infos
    agent_names = list(agent_dicts_dict.keys())
    agent_name_sel = agent_names[0]
    agent_dict = agent_dicts_dict[agent_name_sel]
    USED_TFS = str(agent_dict['general']['USED_TFS']).split(",")

    def agent_dict_process(agent_name_sel):
        global orders_info, df_set, USED_TFS
        agent_dict = agent_dicts_dict[agent_name_sel]
        USED_TFS = str(agent_dict['general']['USED_TFS']).split(",")

        # Process orders
        orders_info = agent_dict['orders_info']

        # Process df_set
        if 'df_sets' in agent_dict.keys():
            df_set = agent_dict['df_sets'][0]
        else:
            df_set = read_df_sets(agent_dict)[0]

        for didx, df in enumerate(df_set):
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df.columns = df.columns.str.lower()
            df.index.name = 'time'
            df = df.reset_index()
            df_set[didx] = df

    def agent_selection(chart):
        global agent_name_sel, lines
        agent_name_sel = chart.topbar['agent'].value
        agent_dict_process(agent_name_sel)

        # Remove markers & lines
        for line in lines:
            line.delete()
        lines = []
        chart.clear_markers()

        # Draw
        chart.set(df_set[-1], True)
        draw_orders(chart)

    def timeframe_selection(chart):
        global lines
        tf_sel = chart.topbar['timeframe'].value

        if tf_sel not in USED_TFS:
            print(tf_sel + " Not in its USED TF " + str(USED_TFS))
        else:
            # Remove markers & lines
            for line in lines:
                line.delete()
            lines = []
            chart.clear_markers()

            # Draw
            chart.set(df_set[USED_TFS.index(tf_sel)], True)
            draw_orders(chart)

    def draw_orders(chart):
        global lines
        lines = []
        markers = []
        for _, order in orders_info.iterrows():
            # draw hanging orders using trend line
            if order['order_type']!=0:
                color = 'green' if order['order_type'] == 1 else 'red'
                lines.append(chart.trend_line(start_time = order['pending_date'], start_value = order['price'],
                                              end_time = order['exec_date'], end_value = order['price'],
                                              line_color = color, width = 2, style = 'dashed'))

            # Draw executed point by ugly marker
            if order['cancel']!=1:
                position = 'below' if order['exec_base'] > 0 else 'above'
                shape = 'arrow_up' if order['exec_base'] > 0 else 'arrow_down'
                color = 'green' if order['exec_base'] > 0 else 'red'
                text = 'open' if order['base'] != 0 else 'close'
                marker = dict(time = order['exec_date'], position = position,
                              shape = shape, color = color, text = text)
                markers.append(marker)

        chart.marker_list(markers)

    # Initialize chart with toolbox and legend
    chart = Chart(toolbox=True, width=1600, height=800)
    chart.legend(True)

    # Add agent selection topbar
    chart.topbar.menu('agent', agent_names, default=agent_name_sel,
                            func=agent_selection)

    # Add timeframe selection topbar
    chart.topbar.switcher('timeframe', used_tfs_all, default=USED_TFS[-1],
                            func=timeframe_selection)

    # Draw default chart
    lines = []
    agent_selection(chart)
    chart.show(block=True)

    # Test
    # chart.exit()

