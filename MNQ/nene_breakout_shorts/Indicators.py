#
# Python Script with indicators 
# calculations
#
# (c) Andres Estrada Cadavid
# QSociety

import pandas as pd
import numpy as np

class Indicators():
    def SMA(self, data, period):
        sma = data.rolling(period).mean()
        return sma
    
    def EMA(self, data, period):
        ema = data.ewm(span=period, min_periods=period, adjust=False).mean()
        return ema

    def MACD(self, data, slow_period=26, fast_period=12, signal_period=9, type='macd'):
        macd = self.EMA(data=data, period=fast_period) - self.EMA(data=data, period=slow_period)
        signal = self.EMA(data=macd, period=signal_period)
        if type == 'macd':
            indicator = macd
        elif type == 'signal':
            indicator = signal
        elif type == 'histogram':
            indicator = macd - signal
        return indicator
    
    def RSI(self, data, period=14):
        deltas = data.diff()
        seed = deltas[:period+1]
        up = seed[seed>=0].sum()/period
        down = -seed[seed<0].sum()/period
        rs = up/down
        rsi = np.zeros(len(deltas))
        rsi[period] = 100 - 100/(rs+1)

        for i in range(period+1,len(deltas)):
            delta = deltas[i]
            if delta >= 0:
                upval = delta; downval = 0
            else:
                upval = 0; downval = abs(delta)
            up = (up*(period-1) + upval)/period
            down = (down*(period-1) + downval)/period
            rs = up/down
            rsi[i] = 100 - (100/(rs+1))
        return rsi
    
    '''def ATR(self, data, period=14):
        atr = pd.concat([(data.high - data.low), abs(data.close.shift(1)-data.high), abs(data.close.shift(1)-data.low)],axis=1) \
                                .max(axis=1).dropna().rolling(period).mean()
        return atr'''
    
    def ATR(self, data, period=14, ma_type='SMA'):
        tr = pd.concat([(data.high - data.low), abs(data.close.shift(1)-data.high), abs(data.close.shift(1)-data.low)],axis=1) \
                                    .max(axis=1).dropna()
        atr = (lambda t: self.SMA(tr, period) if t=='SMA' else self.EMA(tr, period))(ma_type)
        return atr

    def BB(self, data, period=20, multiplier=2, type='midline'):
        std = data.rolling(period).std()
        midline = data.rolling(period).mean()
        if type == 'midline':
            indicator = midline
        elif type == 'upper':
            indicator = midline + multiplier*std
        elif type == 'lower':
            indicator = midline - multiplier*std
        return indicator
    
    def LR(self, data, period):
        sum_per = sum(range(1, period+1))
        sum_per_w = sum([x**2 for x in range(1,period+1)])
        sum_closes = data.rolling(period).sum()
        sum_closes_w = data.rolling(period).apply(self.sum_w, raw=True)
        b = ((period * sum_closes_w) - (sum_per * sum_closes)) / ((period * sum_per_w) - (sum_per**2))
        a = (sum_closes - (b*sum_per)) / period
        LR = a + b * period
        return LR
    
    def Bollinger_Bands(self, data, ma_type='SMA', period=20, dev=2):
        midline = (lambda t: self.SMA(data, period) if t == 'SMA' else self.EMA(data, period))(ma_type)
        std = data.rolling(period).std()
        upper = midline + (dev*std)
        lower = midline - (dev*std)
        return midline, upper, lower
    
    def keltner_channel(self, data, ma_type='EMA', period=10, mult=1):
        midline = (lambda t: self.SMA(data.close, period) if t == 'SMA' else self.EMA(data.close, period))(ma_type)
        atr = self.ATR(data, period)
        upper_band = midline + (mult*atr)
        lower_band = midline - (mult*atr)
        return midline, upper_band, lower_band
    
    def ADX(self, data, period=14, ma_type='EMA'):
        up_move =  data.high - data.high.shift(1)
        down_move = data.low.shift(1) - data.low
        
        plus_dm_condition = (up_move > down_move) & (up_move > 0)
        minus_dm_condition = (down_move > up_move) & (down_move > 0)
        plus_dm = abs(plus_dm_condition * up_move)
        minus_dm = abs(minus_dm_condition * down_move)
        
        plus_di = 100*self.EMA(plus_dm, period)/self.ATR(data, period, ma_type='EMA')
        minus_di = 100*self.EMA(minus_dm, period)/self.ATR(data, period, ma_type='EMA')
        
        adx = 100*self.EMA(abs(plus_di-minus_di)/(plus_di+minus_di), period)
        
        return adx, plus_di, minus_di

    def sum_w(self, values):
        return sum([(i+1) * values[i] for i in range(len(values))])
    
    def update_indicators(self, old_series, new_series):
        return pd.concat([old_series.iloc[:-2], new_series.loc[old_series.index[-2]:]])