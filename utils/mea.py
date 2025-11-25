import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import tkinter as tk
import numpy as np
import matplotlib.dates as mpl_dates


class MEAPlotter:
    def __init__(self, agent_dicts_dict_or_agent_dict, suffix='atr', sample_rate=1.0):
        # Visual style
        plt.style.use('dark_background')
        self.plot_color = '#007ACC'
        self.suffix = suffix

        # Build plot_info directly
        if isinstance(agent_dicts_dict_or_agent_dict, dict) and 'general' in agent_dicts_dict_or_agent_dict:
            orders_info = agent_dicts_dict_or_agent_dict['orders_info']
            orders_info_noc = orders_info.loc[~(orders_info['cancel'] == 1)].reset_index(drop=True)
            orders_info_noc_pair = orders_info_noc.groupby('position_cnt').filter(lambda x: len(x) >= 2)
            orders_info_noc_close_pos = orders_info_noc_pair.loc[orders_info_noc_pair['base'] == 0].copy()

            columns_to_extract = [col for col in orders_info.columns if col.startswith('mea_') and col.endswith(f'_{suffix}')]
            plot_info = orders_info_noc_close_pos[columns_to_extract].copy()
            plot_info['position_cnt'] = orders_info_noc_close_pos['position_cnt']
            plot_info['exec_date'] = orders_info_noc_close_pos['exec_date']
            plot_info = plot_info.reindex(columns=['position_cnt'] + ['exec_date'] + columns_to_extract)
            plot_info.columns = [col.replace('mea_', '').replace(f'_{suffix}', '') for col in plot_info.columns]
        else:
            plot_info = pd.DataFrame()
            for agent_dict in agent_dicts_dict_or_agent_dict.values():
                orders_info = agent_dict['orders_info']
                orders_info_noc = orders_info.loc[~(orders_info['cancel'] == 1)].reset_index(drop=True)
                orders_info_noc_pair = orders_info_noc.groupby('position_cnt').filter(lambda x: len(x) >= 2)
                orders_info_noc_close_pos = orders_info_noc_pair.loc[orders_info_noc_pair['base'] == 0].copy()

                columns_to_extract = [col for col in orders_info.columns if col.startswith('mea_') and col.endswith(f'_{suffix}')]
                plot_info_tmp = orders_info_noc_close_pos[columns_to_extract].copy()
                plot_info_tmp['position_cnt'] = orders_info_noc_close_pos['position_cnt']
                plot_info_tmp = plot_info_tmp.reindex(columns=['position_cnt'] + columns_to_extract)
                plot_info_tmp.columns = [col.replace('mea_', '').replace(f'_{suffix}', '') for col in plot_info_tmp.columns]
                plot_info = pd.concat([plot_info, plot_info_tmp], ignore_index=True)

            plot_info['position_cnt'] = plot_info.index

        self.plot_info = plot_info.sample(frac=sample_rate).sort_index()

    def _create_scatter_with_hover(self, ax, x_data, y_data, x_label, y_label):
        # Split data based on win/loss
        win_mask = self.plot_info['pnl'] > 0
        loss_mask = ~win_mask

        # Scatter points with different colors
        win_scatter = ax.scatter(x_data[win_mask], y_data[win_mask], alpha=0.4,
                                 c='green', edgecolor='white', linewidth=0.5, s=30, label='Win')
        loss_scatter = ax.scatter(x_data[loss_mask], y_data[loss_mask], alpha=0.4,
                                  c='red', edgecolor='white', linewidth=0.5, s=30, label='Loss')

        annot = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w", alpha=0.8),
                            arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

        def update_annot(ind, scatter_obj):
            pos = scatter_obj.get_offsets()[ind]
            selected_df = self.plot_info[win_mask] if scatter_obj == win_scatter else self.plot_info[loss_mask]
            annot.xy = pos
            text = f"Position: {selected_df.iloc[ind]['position_cnt']}\n{x_label}: {pos[0]:.2f}\n{y_label}: {pos[1]:.2f}"
            annot.set_text(text)

        def hover(event):
            for scatter_obj in [win_scatter, loss_scatter]:
                if event.inaxes == ax:
                    cont, ind = scatter_obj.contains(event)
                    if cont:
                        update_annot(ind["ind"][0], scatter_obj)
                        annot.set_visible(True)
                        ax.figure.canvas.draw_idle()
                        return
            if annot.get_visible():
                annot.set_visible(False)
                ax.figure.canvas.draw_idle()

        ax.figure.canvas.mpl_connect("motion_notify_event", hover)

    def _maybe_show_with_tk(self, fig, plt_on):
        # Render in a Tk window when plt_on == 1, otherwise return silently
        if plt_on == 1:
            def on_closing():
                root.quit()
                root.destroy()

            root = tk.Tk()
            canvas = FigureCanvasTkAgg(fig, master=root)
            canvas.draw()
            canvas.get_tk_widget().pack()
            root.protocol("WM_DELETE_WINDOW", on_closing)
            root.mainloop()

    def _quantile_xlim(self, series, thd):
        # Compute symmetric x limits from quantiles
        return (series.quantile(thd), series.quantile(1 - thd))

    # ------------------------------- Time analysis MAE/MFE methods -------------------------------
    def plot_mae_mfe_time(self, plt_on=1):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        ax1.plot(self.plot_info['exec_date'], self.plot_info['mae'], color='red')
        ax1.set_ylabel('MAE')
        ax2.plot(self.plot_info['exec_date'], self.plot_info['mfe'], color='green')
        ax2.set_ylabel('MFE')
        ax3.plot(self.plot_info['exec_date'], self.plot_info['bmfe'], color='green')
        ax3.set_ylabel('BMFE')
        date_format = mpl_dates.DateFormatter('%Y-%m-%d')
        ax3.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()

        plt.tight_layout()
        self._maybe_show_with_tk(fig, plt_on)
        plt.close()
        return fig

    # ------------------------------- Single-plot methods -------------------------------

    def plot_mae_vs_pnl(self, OUTLIAR_THD=0, plt_on=1):
        # MAE vs PNL scatter
        if plt_on == 0:
            plt.ioff()
        fig, ax = plt.subplots(figsize=(6, 5))
        self._create_scatter_with_hover(ax, self.plot_info['pnl'], self.plot_info['mae'], 'PNL', 'MAE')
        ax.set_title('MAE - PNL', color='white')
        xlim = self._quantile_xlim(self.plot_info['pnl'], OUTLIAR_THD)
        ax.set_xlim(xlim)
        ax.set_ylim(0, self.plot_info['mae'].quantile(1 - OUTLIAR_THD))
        plt.tight_layout()
        self._maybe_show_with_tk(fig, plt_on)
        plt.close()
        return fig

    def plot_hist_pnl(self, bin_number=30, OUTLIAR_THD=0, plt_on=1):
        # Histogram of PNL
        if plt_on == 0:
            plt.ioff()
        fig, ax = plt.subplots(figsize=(6, 4))
        xlim = self._quantile_xlim(self.plot_info['pnl'], OUTLIAR_THD)
        sns.histplot(self.plot_info['pnl'], ax=ax, kde=True, alpha=0.5,
                     color=self.plot_color, bins=bin_number, binrange=xlim)
        ax.set_title('Count - PNL', color='white')
        ax.set_xlim(xlim)
        plt.tight_layout()
        self._maybe_show_with_tk(fig, plt_on)
        plt.close()
        return fig

    def plot_mfe_vs_pnl(self, mfe_type='bmfe', OUTLIAR_THD=0, plt_on=1):
        # MFE vs PNL scatter
        if plt_on == 0:
            plt.ioff()
        y = self.plot_info[mfe_type]
        fig, ax = plt.subplots(figsize=(6, 5))
        self._create_scatter_with_hover(ax, self.plot_info['pnl'], y, 'PNL', mfe_type.upper())
        ax.set_title(f'{mfe_type.upper()} - PNL', color='white')
        xlim = self._quantile_xlim(self.plot_info['pnl'], OUTLIAR_THD)
        ax.set_xlim(xlim)
        ax.set_ylim(0, y.quantile(1 - OUTLIAR_THD))
        plt.tight_layout()
        self._maybe_show_with_tk(fig, plt_on)
        plt.close()
        return fig

    def plot_hist_mae(self, bin_number=30, OUTLIAR_THD=0, plt_on=1):
        # Histogram of MAE
        if plt_on == 0:
            plt.ioff()
        fig, ax = plt.subplots(figsize=(6, 4))
        xmax = self.plot_info['mae'].quantile(1 - OUTLIAR_THD)
        bins = np.linspace(0, xmax, bin_number + 1)
        sns.histplot(self.plot_info['mae'], ax=ax, kde=True, alpha=0.5,
                     color=self.plot_color, bins=bins)
        ax.set_title('Count - MAE', color='white')
        ax.set_xlim(0, xmax)
        plt.tight_layout()
        self._maybe_show_with_tk(fig, plt_on)
        plt.close()
        return fig

    def plot_mfe_vs_mae(self, mfe_type='bmfe', OUTLIAR_THD=0, bin_number=30, plt_on=1):
        # MFE vs MAE scatter with 75% guide lines (kept same as original)
        if plt_on == 0:
            plt.ioff()
        fig, ax = plt.subplots(figsize=(6, 5))
        x = self.plot_info['mae']
        y = self.plot_info[mfe_type]
        self._create_scatter_with_hover(ax, x, y, 'MAE', mfe_type.upper())

        # Keep the same dashed lines choice as original (though axes might look swapped)
        ax.axvline(self.plot_info['mae'].quantile(0.75), linestyle='--', color='white')
        ax.axhline(self.plot_info[mfe_type].quantile(0.75), linestyle='--', color='white')

        xmax = max(x.quantile(1 - OUTLIAR_THD), self.plot_info['mfe'].quantile(1 - OUTLIAR_THD)
                   if 'mfe' in self.plot_info.columns else y.quantile(1 - OUTLIAR_THD))
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, xmax)
        ax.set_title(f'{mfe_type.upper()} - MAE', color='white')
        ax.set_xlabel('MAE', color='white')
        ax.set_ylabel(mfe_type.upper(), color='white')
        plt.tight_layout()
        self._maybe_show_with_tk(fig, plt_on)
        plt.close()
        return fig

    def plot_hist_mfe(self, mfe_type='bmfe', bin_number=30, OUTLIAR_THD=0, plt_on=1):
        # Histogram of MFE-type
        if plt_on == 0:
            plt.ioff()
        fig, ax = plt.subplots(figsize=(6, 4))
        xmax = max(self.plot_info['mae'].quantile(1 - OUTLIAR_THD),
                   self.plot_info['mfe'].quantile(1 - OUTLIAR_THD)
                   if 'mfe' in self.plot_info.columns else self.plot_info[mfe_type].quantile(1 - OUTLIAR_THD))
        bins = np.linspace(0, xmax, bin_number + 1)
        sns.histplot(self.plot_info[mfe_type], ax=ax, kde=True, alpha=0.5,
                     color=self.plot_color, bins=bins)
        ax.set_title(f'Count - {mfe_type.upper()}', color='white')
        ax.set_xlim(0, xmax)
        plt.tight_layout()
        self._maybe_show_with_tk(fig, plt_on)
        plt.close()
        return fig

    def plot_hist_winloss_mae(self, bin_number=30, OUTLIAR_THD=0, plt_on=1):
        # Histogram of MAE split by win/loss
        if plt_on == 0:
            plt.ioff()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        xmax = max(self.plot_info['mae'].quantile(1 - OUTLIAR_THD),
                   self.plot_info['mfe'].quantile(1 - OUTLIAR_THD)
                   if 'mfe' in self.plot_info.columns else self.plot_info['mae'].quantile(1 - OUTLIAR_THD))
        bins = np.linspace(0, xmax, bin_number + 1)
        win_mae = self.plot_info['mae'][self.plot_info['pnl'] > 0]
        loss_mae = self.plot_info['mae'][self.plot_info['pnl'] <= 0]
        sns.histplot(win_mae, ax=ax, kde=True, alpha=0.5, color='green', label='Win MAE', bins=bins)
        sns.histplot(loss_mae, ax=ax, kde=True, alpha=0.5, color='red', label='Loss MAE', bins=bins)
        ax.set_xlim(0, xmax)
        ax.set_title('Count - Win/Loss MAE', color='white')
        ax.legend()
        plt.tight_layout()
        self._maybe_show_with_tk(fig, plt_on)
        plt.close()
        return fig

    def plot_hist_mhl(self, bin_number=30, OUTLIAR_THD=0, plt_on=1):
        # Histogram of MHL
        if plt_on == 0:
            plt.ioff()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        xmax = max(self.plot_info['mae'].quantile(1 - OUTLIAR_THD),
                   self.plot_info['mfe'].quantile(1 - OUTLIAR_THD)
                   if 'mfe' in self.plot_info.columns else self.plot_info['mhl'].quantile(1 - OUTLIAR_THD))
        bins = np.linspace(0, xmax, bin_number + 1)
        sns.histplot(self.plot_info['mhl'], ax=ax, kde=True, alpha=0.5,
                     color=self.plot_color, bins=bins)
        ax.set_xlim(0, xmax)
        ax.set_title('Count - MHL', color='white')
        plt.tight_layout()
        self._maybe_show_with_tk(fig, plt_on)
        plt.close()
        return fig

    def plot_mhl_vs_mfe(self, OUTLIAR_THD=0, plt_on=1):
        # MHL vs MFE scatter (labels kept consistent with original figure)
        if plt_on == 0:
            plt.ioff()
        fig, ax = plt.subplots(figsize=(6, 5))
        # Original used x: mfe, y: mhl, then labeled "MHL - GMFE" with GMFE on x label
        self._create_scatter_with_hover(ax, self.plot_info['mfe'], self.plot_info['mhl'], 'MHL', 'GMFE')
        xmax = max(self.plot_info['mae'].quantile(1 - OUTLIAR_THD),
                   self.plot_info['mfe'].quantile(1 - OUTLIAR_THD))
        ax.plot((0, xmax), (0, xmax), linestyle='--', color='white', linewidth=1)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, xmax)
        ax.set_title('MHL - GMFE', color='white')
        ax.set_xlabel('GMFE', color='white')
        ax.set_ylabel('MHL', color='white')
        plt.tight_layout()
        self._maybe_show_with_tk(fig, plt_on)
        plt.close()
        return fig

    def plot_hist_winloss_mfe(self, OUTLIAR_THD=0, bin_number=30, plt_on=1):
        # Histogram of MFE split by win/loss (named GMFE in original titles)
        if plt_on == 0:
            plt.ioff()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        xmax = max(self.plot_info['mae'].quantile(1 - OUTLIAR_THD),
                   self.plot_info['mfe'].quantile(1 - OUTLIAR_THD))
        bins = np.linspace(0, xmax, bin_number + 1)
        win_mfe = self.plot_info['mfe'][self.plot_info['pnl'] > 0]
        loss_mfe = self.plot_info['mfe'][self.plot_info['pnl'] <= 0]
        sns.histplot(win_mfe, ax=ax, kde=True, alpha=0.5, color='green', label='Win MFE', bins=bins)
        sns.histplot(loss_mfe, ax=ax, kde=True, alpha=0.5, color='red', label='Loss MFE', bins=bins)
        ax.set_xlim(0, xmax)
        ax.set_title('Count - GMFE', color='white')
        ax.legend()
        plt.tight_layout()
        self._maybe_show_with_tk(fig, plt_on)
        plt.close()
        return fig

    # ------------------------------- Aggregations -------------------------------

    def compute_mea_df(self, mfe_type='bmfe'):
        # Compute quantile summary dataframe
        pnl_q1 = self.plot_info['pnl'].quantile(0.25)
        pnl_q2 = self.plot_info['pnl'].quantile(0.50)
        pnl_q3 = self.plot_info['pnl'].quantile(0.75)
        mae_q1 = self.plot_info['mae'].quantile(0.25)
        mae_q2 = self.plot_info['mae'].quantile(0.50)
        mae_q3 = self.plot_info['mae'].quantile(0.75)
        mfe_q1 = self.plot_info[mfe_type].quantile(0.25)
        mfe_q2 = self.plot_info[mfe_type].quantile(0.50)
        mfe_q3 = self.plot_info[mfe_type].quantile(0.75)
        mea_df = pd.DataFrame([{
            'suffix': self.suffix,
            'pnl_q1': pnl_q1, 'pnl_q2': pnl_q2, 'pnl_q3': pnl_q3,
            'mae_q1': mae_q1, 'mae_q2': mae_q2, 'mae_q3': mae_q3,
            'mfe_q1': mfe_q1, 'mfe_q2': mfe_q2, 'mfe_q3': mfe_q3,
        }])
        return mea_df

    def plot_all(self, mfe_type='bmfe', OUTLIAR_THD=0, bin_number=30, plt_on=1):
        # Reproduce the original full dashboard in a 10x3 grid
        if plt_on == 0:
            plt.ioff()

        mea_df = pd.DataFrame(columns=[
            'suffix', 'pnl_q1', 'pnl_q2', 'pnl_q3',
            'mae_q1', 'mae_q2', 'mae_q3',
            'mfe_q1', 'mfe_q2', 'mfe_q3'
        ])

        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(10, 3, height_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        # Left column (tall-short-tall)
        ax1 = fig.add_subplot(gs[:4, 0])
        self._create_scatter_with_hover(ax1, self.plot_info['pnl'], self.plot_info['mae'], 'PNL', 'MAE')
        ax1.set_title('MAE - PNL', color='white')
        common_xlim_left = self._quantile_xlim(self.plot_info['pnl'], OUTLIAR_THD)
        ax1.set_xlim(common_xlim_left)
        ax1.set_ylim(0, self.plot_info['mae'].quantile(1 - OUTLIAR_THD))

        ax2 = fig.add_subplot(gs[4:6, 0])
        sns.histplot(self.plot_info['pnl'], ax=ax2, kde=True, alpha=0.5,
                     color=self.plot_color, bins=bin_number, binrange=common_xlim_left)
        ax2.set_title('Count - PNL', color='white')
        ax2.set_xlim(common_xlim_left)

        ax3 = fig.add_subplot(gs[6:, 0])
        self._create_scatter_with_hover(ax3, self.plot_info['pnl'], self.plot_info[mfe_type], 'PNL', mfe_type.upper())
        ax3.set_title(f'{mfe_type.upper()} - PNL', color='white')
        ax3.set_xlim(common_xlim_left)
        ax3.set_ylim(0, self.plot_info[mfe_type].quantile(1 - OUTLIAR_THD))

        # Mid column (short-tall-short)
        xmax_mid = max(self.plot_info['mae'].quantile(1 - OUTLIAR_THD),
                       self.plot_info['mfe'].quantile(1 - OUTLIAR_THD)
                       if 'mfe' in self.plot_info.columns else self.plot_info[mfe_type].quantile(1 - OUTLIAR_THD))
        fixed_bins = np.linspace(0, xmax_mid, bin_number + 1)

        ax4 = fig.add_subplot(gs[:3, 1])
        sns.histplot(self.plot_info['mae'], ax=ax4, kde=True, alpha=0.5,
                     color=self.plot_color, bins=fixed_bins)
        ax4.set_xlim(0, xmax_mid)
        ax4.set_title('Count - MAE', color='white')

        ax5 = fig.add_subplot(gs[3:7, 1])
        self._create_scatter_with_hover(ax5, self.plot_info['mae'], self.plot_info[mfe_type], 'MAE', mfe_type.upper())
        ax5.axvline(self.plot_info['mae'].quantile(0.75), linestyle='--', color='white')
        ax5.axhline(self.plot_info[mfe_type].quantile(0.75), linestyle='--', color='white')
        ax5.set_xlim(0, xmax_mid)
        ax5.set_ylim(0, xmax_mid)
        ax5.set_title(f'{mfe_type.upper()} - MAE', color='white')
        ax5.set_xlabel('MAE', color='white')
        ax5.set_ylabel(mfe_type.upper(), color='white')

        ax6 = fig.add_subplot(gs[7:, 1])
        sns.histplot(self.plot_info[mfe_type], ax=ax6, kde=True, alpha=0.5,
                     color=self.plot_color, bins=fixed_bins)
        ax6.set_xlim(0, xmax_mid)
        ax6.set_title(f'Count - {mfe_type.upper()}', color='white')

        # Right column (short-short-tall-short)
        ax7 = fig.add_subplot(gs[:2, 2])
        win_mae = self.plot_info['mae'][self.plot_info['pnl'] > 0]
        loss_mae = self.plot_info['mae'][self.plot_info['pnl'] <= 0]
        sns.histplot(win_mae, ax=ax7, kde=True, alpha=0.5, color='green', label='Win MAE', bins=fixed_bins)
        sns.histplot(loss_mae, ax=ax7, kde=True, alpha=0.5, color='red', label='Loss MAE', bins=fixed_bins)
        ax7.set_xlim(0, xmax_mid)
        ax7.set_title('Count - Win/Loss MAE', color='white')
        ax7.legend()

        ax8 = fig.add_subplot(gs[2:4, 2])
        sns.histplot(self.plot_info['mhl'], ax=ax8, kde=True, alpha=0.5,
                     color=self.plot_color, bins=fixed_bins)
        ax8.set_xlim(0, xmax_mid)
        ax8.set_title('Count - MHL', color='white')

        ax9 = fig.add_subplot(gs[4:8, 2])
        self._create_scatter_with_hover(ax9, self.plot_info['mfe'], self.plot_info['mhl'], 'MHL', 'GMFE')
        ax9.plot((0, xmax_mid), (0, xmax_mid), linestyle='--', color='white', linewidth=1)
        ax9.set_xlim(0, xmax_mid)
        ax9.set_ylim(0, xmax_mid)
        ax9.set_title('MHL - GMFE', color='white')
        ax9.set_xlabel('GMFE', color='white')
        ax9.set_ylabel('MHL', color='white')

        ax10 = fig.add_subplot(gs[8:, 2])
        win_mfe = self.plot_info['mfe'][self.plot_info['pnl'] > 0]
        loss_mfe = self.plot_info['mfe'][self.plot_info['pnl'] <= 0]
        sns.histplot(win_mfe, ax=ax10, kde=True, alpha=0.5, color='green', label='Win MFE', bins=fixed_bins)
        sns.histplot(loss_mfe, ax=ax10, kde=True, alpha=0.5, color='red', label='Loss MFE', bins=fixed_bins)
        ax10.set_xlim(0, xmax_mid)
        ax10.set_title('Count - GMFE', color='white')
        ax10.legend()

        fig.suptitle(f"Visualization for Suffix: {self.suffix}, MFE Type: {mfe_type}",
                     fontsize=16, color='white')
        plt.tight_layout()

        self._maybe_show_with_tk(fig, plt_on)
        plt.close()
        return fig
