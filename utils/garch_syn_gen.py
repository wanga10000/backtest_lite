import pandas as pd
import numpy as np
from arch import arch_model
from scipy.stats import genextreme
import warnings
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


class GarchPriceGen:
    def __init__(self, ohlcv_df, trend_on=0):
        self.trend_on = trend_on
        self.close_series = ohlcv_df['Close']
        self.initial_price = ohlcv_df['Open'][0]

        if trend_on == 1:
            # Extract trend using ARIMA model on returns
            self.trend_returns = self._extract_trend_arima()
            # Calculate residual returns (detrended returns)
            all_returns = self.close_series.pct_change().dropna()
            self.returns = all_returns - self.trend_returns
        else:
            self.trend_returns = None
            self.arima_model = None
            self.returns = self.close_series.pct_change().dropna()

        # Fit GARCH model for volatility
        self.garch_model = arch_model(self.returns,
                                     vol='GARCH',
                                     p=1,
                                     q=1,
                                     dist='skewt',
                                     rescale=True,
                                     )
        self.garch_results = self.garch_model.fit(disp='off')
        self.scale = self.garch_results.scale
        self.max_return = self.returns.abs().quantile(0.95)
        vol = self.garch_results.conditional_volatility
        vol = pd.concat((pd.Series(vol[0]), vol))

        # Fit high low distribution
        if trend_on == 1:
            # Use combined returns (residual + trend) for high/low distribution fitting
            combined_original_returns = self.close_series.pct_change().dropna()
            close_for_hl = self.close_series.values
        else:
            close_for_hl = ohlcv_df['Close'].values

        open1 = ohlcv_df['Open'].values
        prev_close = np.concatenate([[np.nan], close_for_hl[:-1]])
        delta = (close_for_hl - prev_close)
        high_rel = (ohlcv_df['High'].values - open1)/vol
        low_rel = -(ohlcv_df['Low'].values - open1)/vol

        tmp = pd.DataFrame({
            'delta': delta,
            'high': high_rel,
            'low': low_rel
        }).dropna()

        tmp['group'] = np.where(tmp['delta'] > 0, 'up', 'down')

        dists = {}
        for g in ['up', 'down']:
            sub = tmp[tmp['group'] == g]
            high_params = genextreme.fit(sub['high'])
            high_params = high_params + (sub['high'].quantile(0.99),)
            low_params = genextreme.fit(sub['low'])
            low_params = low_params + (sub['low'].quantile(0.99),)
            dists[g] = {
                'high': high_params,
                'low':  low_params
            }
        self.dists = dists

        # Syn real hist compare
        # for g in [('up', 'high'), ('up', 'low'), ('down', 'high'), ('down', 'low')]:
        #     params = dists[g[0]][g[1]]
        #     data = tmp[tmp['group'] == g[0]][g[1]]
        #     synthetic = genextreme.rvs(params[0], params[1], params[2], size=len(data))
        #     synthetic = synthetic.clip(max = params[3])
        #     min_val = np.min(data)
        #     max_val = np.max(data)
        #     plt.figure(figsize=(16, 6))
        #     plt.hist(data, bins=60, range=(min_val, max_val), density=True, alpha=0.5, edgecolor='black', label='Original Data')
        #     plt.hist(synthetic, bins=60, range=(min_val, max_val), density=True, alpha=0.5, edgecolor='gray', label='Fitted Gamma Samples')
        #     plt.legend()


    def _extract_trend_arima(self):
        returns = self.close_series.pct_change().dropna()
        model = ARIMA(returns.values, order=(2,0,2))
        stats_mdl = model.fit()
        # print(stats_mdl.summary())

        # Get fitted values as trend
        trend = pd.Series(stats_mdl.fittedvalues, index=self.close_series.index[1:])

        return trend


    def generate_close(self, num_series=10):
        # Get original series length and initial price
        series_length = len(self.close_series)
        initial_price = self.close_series.iloc[0]

        # Create DataFrame to store generated series
        synthetic_df = pd.DataFrame(index=self.close_series.index)

        for i in range(num_series):
            # Sim return
            sims = self.garch_model.simulate(self.garch_results.params, nobs=series_length)
            sim_returns = sims['data'] / self.scale
            sim_returns = sim_returns.clip(lower=-self.max_return, upper=self.max_return)

            # Add trend back if trend_on is enabled
            if self.trend_on == 1:
                # Calculate trend difference from initial level
                combined_returns = sim_returns.values[1:] + self.trend_returns
                prices = initial_price * np.cumprod(1 + combined_returns)
                prices = np.concatenate(([initial_price], prices))
                prices = pd.Series(prices, index=self.close_series.index)
            else:
                prices = initial_price * np.cumprod(1 + sim_returns.values)

            # Add generated series to DataFrame
            series_name = f'synthetic_{i+1}'
            synthetic_df[series_name] = prices

        return synthetic_df


    def generate_ohlcv(self):
        # Get original series length and initial price
        series_length = len(self.close_series)
        initial_price = self.close_series.iloc[0]

        # Sim return for open and close
        sims = self.garch_model.simulate(self.garch_results.params, nobs=series_length)
        sim_returns = sims['data'] / self.scale
        sim_returns = sim_returns.clip(lower=-self.max_return, upper=self.max_return)
        sim_vols = sims['volatility'].values

        # Add trend back if trend_on is enabled
        if self.trend_on == 1:
            # Calculate trend difference from initial level
            combined_returns = sim_returns.values[1:] + self.trend_returns
            prices = initial_price * np.cumprod(1 + combined_returns)
            prices = np.concatenate(([initial_price], prices))
            closes = pd.Series(prices, index=self.close_series.index)
        else:
            closes = initial_price * np.cumprod(1 + sim_returns.values)

        opens = np.insert(closes[0:-1], 0, initial_price)

        # Sim high and close
        highs = np.zeros(series_length)
        lows = np.zeros(series_length)
        prev_closes = np.insert(closes[:-1], 0, initial_price)
        directions = np.where(closes - prev_closes > 0, 'up', 'down')

        for i in range(series_length):
            direction = directions[i]
            high_params = self.dists[direction]['high']
            low_params = self.dists[direction]['low']

            high_rel = genextreme.rvs(high_params[0], high_params[1], high_params[2], size=1)[0]
            low_rel = genextreme.rvs(low_params[0], low_params[1], low_params[2], size=1)[0]

            high_rel = min(high_rel, high_params[3])
            low_rel = min(low_rel, low_params[3])

            highs[i] = opens[i] + high_rel * sim_vols[i]
            lows[i] = opens[i] - low_rel * sim_vols[i]

            highs[i] = max(highs[i], opens[i], closes[i])
            lows[i] = min(lows[i], opens[i], closes[i])

        return pd.DataFrame({'Date': self.close_series.index, 'Open': opens, 'High': highs, 'Low': lows, 'Close': closes})


def calculate_pf(orders_info):
    positive_sum = orders_info[orders_info['pnl'] > 0]['pnl'].sum()
    negative_sum = orders_info[orders_info['pnl'] < 0]['pnl'].sum()

    if negative_sum == 0:
        return float('inf') if positive_sum > 0 else 0

    return positive_sum / abs(negative_sum)