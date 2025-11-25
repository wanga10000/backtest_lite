import numpy as np
import pandas as pd
from ta.volatility import AverageTrueRange
from ta.trend import EMAIndicator, CCIIndicator, AroonIndicator, ADXIndicator
from function.func_lib import supertrend, NATR_improved, efficiency_ratio


class filter_tester:
    def __init__(self):
        pass


    def load_strategy_cfg(self, strategy_cfg, USED_TFS):
        self.USED_TFS = USED_TFS
        self.main_tf_idx = USED_TFS.index(str(strategy_cfg['MAIN_TF']))
        self.TESTER_COARSES = list(map(int, str(strategy_cfg['TESTER_COARSES']).split(",")))
        self.TESTER_FINES = list(map(int, str(strategy_cfg['TESTER_FINES']).split(",")))
        self.TESTER_REVERSE = int(strategy_cfg['TESTER_REVERSE'])


    def load_df_sets(self, df_sets):
        self.df_sets = df_sets
        self.declare_df()


    def declare_df(self):
        # Main symbol df handle
        df_set = self.df_sets[0]

        # Main tf handle
        self.df = df_set[self.main_tf_idx]

        # Day df handle
        self.NEED_DAY_COMB = {(0, 8), (2, 1)}
        for fidx, coarse in enumerate(self.TESTER_COARSES):
            fine = self.TESTER_FINES[fidx]
            if ('1d' not in self.USED_TFS) and ((coarse, fine) in self.NEED_DAY_COMB):
                raise ValueError("Day df should be included")
            elif (coarse, fine) in self.NEED_DAY_COMB:
                self.day_tf_idx = self.USED_TFS.index('1d')
                self.df_d = df_set[self.day_tf_idx]
                break


    def indicator_calc(self):
        for fidx, coarse in enumerate(self.TESTER_COARSES):
            fine = self.TESTER_FINES[fidx]
            # C0. Trend filter (long period)
            # C1. Trend filter (small period)
            # C2. Momentum filter
            # C3. Volatility filter
            # C4. Volume filter
            # C5. Noise filter
            if coarse==0:
                # F0. close><EMA(200)
                # F1. CCI(100)><+-THD
                # F2. SUPERTREND(200, 3)
                # F3. AROON(50)
                # F4. Double EMA(50)(100)
                # F5. Triple EMA(50)(100)(200)
                # F6. Double CCI(50)(100)
                # F7. ADX(100) DI+><DI-
                # F8. closed>EMAd(14)
                if fine==0:
                    self.df['TEST_EMA'] = EMAIndicator(close=self.df['Close'], window=200).ema_indicator()
                elif fine==1:
                    self.df['TEST_CCI'] = CCIIndicator(high=self.df['High'],
                                                       low=self.df['Low'],
                                                       close=self.df['Close'],
                                                       window=100).cci()
                elif fine==2:
                    _, _, self.df['TEST_SP'] = supertrend(high=self.df['High'],
                                                          low=self.df['Low'],
                                                          close=self.df['Close'],
                                                          atr_len=200,
                                                          atr_factor=4)
                elif fine==3:
                    aroon = AroonIndicator(close=self.df['Close'], window=50)
                    self.df['TEST_ARO_UP'] = aroon.aroon_up()
                    self.df['TEST_ARO_DOWN'] = aroon.aroon_down()
                elif fine==4:
                    self.df['TEST_SEMA'] = EMAIndicator(close=self.df['Close'], window=50).ema_indicator()
                    self.df['TEST_LEMA'] = EMAIndicator(close=self.df['Close'], window=100).ema_indicator()
                elif fine==5:
                    self.df['TEST_SEMA'] = EMAIndicator(close=self.df['Close'], window=50).ema_indicator()
                    self.df['TEST_MEMA'] = EMAIndicator(close=self.df['Close'], window=100).ema_indicator()
                    self.df['TEST_LEMA'] = EMAIndicator(close=self.df['Close'], window=200).ema_indicator()
                elif fine==6:
                    self.df['TEST_SCCI'] = CCIIndicator(high=self.df['High'],
                                                        low=self.df['Low'],
                                                        close=self.df['Close'],
                                                        window=50).cci()
                    self.df['TEST_LCCI'] = CCIIndicator(high=self.df['High'],
                                                        low=self.df['Low'],
                                                        close=self.df['Close'],
                                                        window=100).cci()
                elif fine==7:
                    adx = ADXIndicator(high=self.df['High'],
                                       low=self.df['Low'],
                                       close=self.df['Close'],
                                       window=100)
                    self.df['TEST_DIP'] = adx.adx_pos()
                    self.df['TEST_DIN'] = adx.adx_neg()
                elif fine==8:
                    self.df_d['TEST_EMA'] = EMAIndicator(close=self.df_d['Close'], window=14).ema_indicator()
                elif fine>=9:
                    raise ValueError("Filter test fine overflow")
                
            elif coarse==1:
                # F0. close><EMA(20)
                # F1. CCI(20)><+-THD
                # F2. SUPERTREND(20, 3)
                # F3. AROON(10)
                # F4. ADX(14) DI+><DI- + adx>25
                if fine==0:
                    self.df['TEST_EMA'] = EMAIndicator(close=self.df['Close'], window=20).ema_indicator()
                elif fine==1:
                    self.df['TEST_CCI'] = CCIIndicator(high=self.df['High'],
                                                       low=self.df['Low'],
                                                       close=self.df['Close'],
                                                       window=20).cci()
                elif fine==2:
                    _, _, self.df['TEST_SP'] = supertrend(high=self.df['High'],
                                                          low=self.df['Low'],
                                                          close=self.df['Close'],
                                                          atr_len=20,
                                                          atr_factor=3)
                elif fine==3:
                    aroon = AroonIndicator(close=self.df['Close'], window=10)
                    self.df['TEST_ARO_UP'] = aroon.aroon_up()
                    self.df['TEST_ARO_DOWN'] = aroon.aroon_down()
                elif fine==4:
                    adx = ADXIndicator(high=self.df['High'],
                                       low=self.df['Low'],
                                       close=self.df['Close'],
                                       window=14)
                    self.df['TEST_DIP'] = adx.adx_pos()
                    self.df['TEST_DIN'] = adx.adx_neg()
                elif fine>=5:
                    raise ValueError("Filter test fine overflow")

            elif coarse==2:
                # F0. close[n] >< close[n-50]
                # F1. closed[n] >< opend[n]
                # F2. ADX(14) [n] > ADX(14)[n-10]
                # F3. ADX(14) [n] >< 20
                if fine==2 or fine==3:
                    adx = ADXIndicator(high=self.df['High'],
                                       low=self.df['Low'],
                                       close=self.df['Close'],
                                       window=14)
                    self.df['TEST_ADX'] = adx.adx()
                elif fine>=4:
                    raise ValueError("Filter test fine overflow")

            elif coarse==3:
                # F0. Double ATR(100)(200)
                # F1. Double ATR(25)(50)
                # F2. NATR(25,50)
                # F3. STD(25)><ATR(25)
                if fine==0:
                    self.df['TEST_SATR'] = AverageTrueRange(high=self.df['High'],
                                                            low=self.df['Low'],
                                                            close=self.df['Close'],
                                                            window=100).average_true_range()
                    self.df['TEST_LATR'] = AverageTrueRange(high=self.df['High'],
                                                            low=self.df['Low'],
                                                            close=self.df['Close'],
                                                            window=200).average_true_range()
                elif fine==1:
                    self.df['TEST_SATR'] = AverageTrueRange(high=self.df['High'],
                                                            low=self.df['Low'],
                                                            close=self.df['Close'],
                                                            window=25).average_true_range()
                    self.df['TEST_LATR'] = AverageTrueRange(high=self.df['High'],
                                                            low=self.df['Low'],
                                                            close=self.df['Close'],
                                                            window=50).average_true_range()
                elif fine==2:
                    self.df['TEST_NATR'] = NATR_improved(high=self.df['High'],
                                                         low=self.df['Low'],
                                                         close=self.df['Close'],
                                                         window=25,
                                                         nor_window=50)
                elif fine==3:
                    self.df['TEST_ATR'] = AverageTrueRange(high=self.df['High'],
                                                           low=self.df['Low'],
                                                           close=self.df['Close'],
                                                           window=25).average_true_range()

                    self.df['TEST_STD'] = self.df['Close'].rolling(25).std()

                elif fine>=4:
                    raise ValueError("Filter test fine overflow")

            elif coarse==4:
                # F0. volume><EMA(50)
                # F1. volume><EMA(100)
                # F2. volume><EMA(25)
                # F3. EMA(25)><EMA(50)
                if fine==0:
                    self.df['TEST_VEMA'] = EMAIndicator(close=self.df['Volume'], window=50).ema_indicator()
                elif fine==1:
                    self.df['TEST_VEMA'] = EMAIndicator(close=self.df['Volume'], window=100).ema_indicator()
                elif fine==2:
                    self.df['TEST_VEMA'] = EMAIndicator(close=self.df['Volume'], window=25).ema_indicator()
                elif fine==3:
                    self.df['TEST_SVEMA'] = EMAIndicator(close=self.df['Volume'], window=25).ema_indicator()
                    self.df['TEST_LVEMA'] = EMAIndicator(close=self.df['Volume'], window=50).ema_indicator()
                elif fine>=4:
                    raise ValueError("Filter test fine overflow")

            elif coarse==5:
                # F0. efficiency ratio<>0.3
                if fine==0:
                    self.df['TEST_ER'] = efficiency_ratio(close=self.df['Close'], er_len=20)
                elif fine>=1:
                    raise ValueError("Filter test fine overflow")

            else:
                raise ValueError("Filter test coarse overflow")


    def check_filter(self, pre_side, kidx_sets):
        idx = kidx_sets[0][self.main_tf_idx]
        side = pre_side if self.TESTER_REVERSE==0 else -pre_side
        conditions = []

        for fidx, coarse in enumerate(self.TESTER_COARSES):
            fine = self.TESTER_FINES[fidx]
            
            if (coarse, fine) in self.NEED_DAY_COMB:
                idx_d = kidx_sets[0][self.day_tf_idx]
                
            # C0. Trend filter (long period)
            # C1. Trend filter (small period)
            # C2. Momentum filter
            # C3. Volatility filter
            # C4. Volume filter
            # C5. Noise filter
            if coarse==0:
                # F0. close><EMA(200)
                # F1. CCI(100)><+-THD
                # F2. SUPERTREND(200, 4)
                # F3. AROON(50)
                # F4. Double EMA(50)(100)
                # F5. Triple EMA(50)(100)(200)
                # F6. Double CCI(50)(100)
                # F7. ADX(100) DI+><DI-
                # F8. closed>EMAd(14)
                if fine==0:
                    if side==1:
                        conditions.append(self.df['Close'][idx]>self.df['TEST_EMA'][idx])
                    else:
                        conditions.append(self.df['Close'][idx]<self.df['TEST_EMA'][idx])
                elif fine==1:
                    if side==1:
                        conditions.append(self.df['TEST_CCI'][idx] > 0)
                    else:
                        conditions.append(self.df['TEST_CCI'][idx] < 0)
                elif fine==2:
                    if side==1:
                        conditions.append(self.df['TEST_SP'][idx]==1)
                    else:
                        conditions.append(self.df['TEST_SP'][idx]==-1)
                elif fine==3:
                    if side==1:
                        conditions.append(self.df['TEST_ARO_UP'][idx] > 50)
                    else:
                        conditions.append(self.df['TEST_ARO_DOWN'][idx] > 50)
                elif fine==4:
                    if side==1:
                        conditions.append(self.df['TEST_SEMA'][idx]>self.df['TEST_LEMA'][idx])
                    else:
                        conditions.append(self.df['TEST_SEMA'][idx]<self.df['TEST_LEMA'][idx])
                elif fine==5:
                    if side==1:
                        conditions.append(self.df['TEST_SEMA'][idx]>self.df['TEST_MEMA'][idx])
                        conditions.append(self.df['TEST_MEMA'][idx]>self.df['TEST_LEMA'][idx])
                    else:
                        conditions.append(self.df['TEST_SEMA'][idx]<self.df['TEST_MEMA'][idx])
                        conditions.append(self.df['TEST_MEMA'][idx]<self.df['TEST_LEMA'][idx])
                elif fine==6:
                    if side==1:
                        conditions.append(self.df['TEST_SCCI'][idx]>self.df['TEST_LCCI'][idx])
                    else:
                        conditions.append(self.df['TEST_SCCI'][idx]<self.df['TEST_LCCI'][idx])
                elif fine==7:
                    if side==1:
                        conditions.append(self.df['TEST_DIP'][idx]>self.df['TEST_DIN'][idx])
                    else:
                        conditions.append(self.df['TEST_DIP'][idx]<self.df['TEST_DIN'][idx])
                elif fine==8:
                    if side==1:
                        conditions.append(self.df_d['Close'][idx_d]>self.df_d['TEST_EMA'][idx_d])
                    else:
                        conditions.append(self.df_d['Close'][idx_d]<self.df_d['TEST_EMA'][idx_d])

            elif coarse==1:
                # F0. close><EMA(20)
                # F1. CCI(20)><+-THD
                # F2. SUPERTREND(20, 3)
                # F3. AROON(10)
                # F4. ADX(14) DI+><DI- + adx>25
                if fine==0:
                    if side==1:
                        conditions.append(self.df['Close'][idx]>self.df['TEST_EMA'][idx])
                    else:
                        conditions.append(self.df['Close'][idx]<self.df['TEST_EMA'][idx])
                elif fine==1:
                    if side==1:
                        conditions.append(self.df['TEST_CCI'][idx] > 0)
                    else:
                        conditions.append(self.df['TEST_CCI'][idx] < -0)
                elif fine==2:
                    if side==1:
                        conditions.append(self.df['TEST_SP'][idx]==1)
                    else:
                        conditions.append(self.df['TEST_SP'][idx]==-1)
                elif fine==3:
                    if side==1:
                        conditions.append(self.df['TEST_ARO_UP'][idx] > 50)
                    else:
                        conditions.append(self.df['TEST_ARO_DOWN'][idx] > 50)
                elif fine==4:
                    if side==1:
                        conditions.append(self.df['TEST_DIP'][idx]>self.df['TEST_DIN'][idx])
                    else:
                        conditions.append(self.df['TEST_DIP'][idx]<self.df['TEST_DIN'][idx])

            elif coarse==2:
                # F0. close[n] >< close[n-50]
                # F1. closed[n] >< opend[n]
                # F2. ADX(14) [n] >< ADX(14)[n-10]
                # F3. ADX(14) [n] >< 20
                if fine==0:
                    if side==1:
                        conditions.append(self.df['Close'][idx]>self.df['Close'][idx-50])
                    else:
                        conditions.append(self.df['Close'][idx]<self.df['Close'][idx-50])
                elif fine==1:
                    if side==1:
                        conditions.append(self.df_d['Close'][idx_d]>self.df_d['Open'][idx_d])
                    else:
                        conditions.append(self.df_d['Close'][idx_d]<self.df_d['Open'][idx_d])
                elif fine==2:
                    if self.TESTER_REVERSE==0:
                        conditions.append(self.df['TEST_ADX'][idx]>self.df['TEST_ADX'][idx-10])
                    else:
                        conditions.append(self.df['TEST_ADX'][idx]<self.df['TEST_ADX'][idx-10])
                elif fine==3:
                    if self.TESTER_REVERSE==0:
                        conditions.append(self.df['TEST_ADX'][idx]>20)
                    else:
                        conditions.append(self.df['TEST_ADX'][idx]<20)

            elif coarse==3:
                # F0. Double ATR(100)(200)
                # F1. Double ATR(25)(50)
                # F2. NATR(25,50)
                # F3. STD(25)><ATR(25)
                if fine==0 or fine==1:
                    if self.TESTER_REVERSE==0:
                        conditions.append(self.df['TEST_SATR'][idx]>self.df['TEST_LATR'][idx])
                    else:
                        conditions.append(self.df['TEST_SATR'][idx]<self.df['TEST_LATR'][idx])
                elif fine==2:
                    if self.TESTER_REVERSE==0:
                        conditions.append(self.df['TEST_NATR'][idx]>0)
                    else:
                        conditions.append(self.df['TEST_NATR'][idx]<=0)
                elif fine==3:
                    if self.TESTER_REVERSE==0:
                        conditions.append(self.df['TEST_STD'][idx]>self.df['TEST_ATR'][idx])
                    else:
                        conditions.append(self.df['TEST_STD'][idx]<self.df['TEST_ATR'][idx])

            elif coarse==4:
                # F0. volume><EMA(50)
                # F1. volume><EMA(100)
                # F2. volume><EMA(25)
                # F3. EMA(25)><EMA(50)
                if fine>=0 and fine<=2:
                    if self.TESTER_REVERSE==0:
                        conditions.append(self.df['Volume'][idx]>self.df['TEST_VEMA'][idx])
                    else:
                        conditions.append(self.df['Volume'][idx]<self.df['TEST_VEMA'][idx])
                elif fine==3:
                    if self.TESTER_REVERSE==0:
                        conditions.append(self.df['TEST_SVEMA'][idx]>self.df['TEST_LVEMA'][idx])
                    else:
                        conditions.append(self.df['TEST_SVEMA'][idx]<self.df['TEST_LVEMA'][idx])

            elif coarse==5:
                # F0. efficiency ratio<>0.3
                if fine==0:
                    if self.TESTER_REVERSE==0:
                        conditions.append(self.df['TEST_ER'][idx]<0.3)
                    else:
                        conditions.append(self.df['TEST_ER'][idx]>0.3)

        return all(conditions)