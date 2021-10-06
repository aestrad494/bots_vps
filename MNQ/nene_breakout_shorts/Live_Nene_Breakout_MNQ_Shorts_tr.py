#
# Python Script with live trading 
# Nene Strategy Class
# 
# (c) Andres Estrada Cadavid
# QSociety

from Live_Class import Live
from Indicators import Indicators
import pyrenko
import numpy as np
import pandas as pd
import scipy.optimize as opt
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from os import path, mkdir
from tzlocal import get_localzone

class LiveNene(Live, Indicators):
    def trailing_stop(self, price_in, trailing, sl, order):
        '''Trailing Stop

        Parameters:
            price_in (float): entry price
            trailing (float): trailing stop level
            sl (float): current stop loss price level
            order (object): order stop loss
        
        Returns:
            float: updated stop loss price level
        '''
        new_sl = sl
        if trailing > 0:
            if self.position > 0:
                if self.x_round(self.data.iloc[-1].high) - price_in >= trailing:
                    if sl < self.x_round(self.data.iloc[-1].high) - trailing:
                        new_sl = round(self.x_round(self.data.iloc[-1].high) - trailing, self.digits)
                        order.auxPrice = new_sl
                        self.ib.placeOrder(self.contract,order)
                        self.print('trailing stop from %5.2f to %5.2f'%(sl, new_sl))
            if self.position < 0:
                if price_in - self.x_round(self.data.iloc[-1].low) >= trailing:
                    if sl > self.x_round(self.data.iloc[-1].low) + trailing:
                        new_sl = round(self.x_round(self.data.iloc[-1].low) + trailing, self.digits)
                        order.auxPrice = new_sl
                        self.ib.placeOrder(self.contract,order)
                        self.print('trailing stop from %5.2f to %5.2f'%(sl, new_sl))
        return self.x_round(new_sl)
    
    def evaluate_renko(self, brick, history, dates, column_name):
        '''Renko qualification

        Parameters:
            brick (float): brick size
            history (list): values (prices) to build renko data
            dates (list): dates of the values (prices) to calculate renko data
            column_name (str): column name to be evaluated

        Returns:
            float: renko evaluation parameter according to column_name
        '''
        renko_obj = pyrenko.renko()
        renko_obj.set_brick_size(brick_size = brick)
        renko_obj.build_history(prices=history, dates=dates)
        return renko_obj.evaluate()[column_name]
    
    def get_brick_size(self, data):
        '''Get brick size according to ATR method

        Parameters:
            data (DataFrame): data to build renko

        Returns:
            float: brick size
        '''
        # Getting bounds by ATR
        atr = self.ATR(data=data, period = 14)
        atr = atr[np.isnan(atr) == False]
        
        # Brick size maximization
        opt_bs = opt.fminbound(lambda x: -self.evaluate_renko(brick = x, history = data.close, dates=data.index,
                                        column_name = 'score'), np.min(atr), np.max(atr), disp=0)
        
        return opt_bs

    def optimal_brick(self, data, stop, contracts):
        '''Calculates the optimal brick size

        Parameters:
            data (DataFrame): data to calculate optimal brick size 
            stop (int): number of renko bars that defines stop level
        
        Returns:
            float: optimal brick size
            float: optimal brick size score
        '''
        temporalities = ['1Min', '2Min', '3Min', '4Min', '5Min', '6Min', '10Min', '12Min', '15Min', '20Min', '30Min']
        total_results = []
        for tempo in temporalities:
            data_res = self.resampler(data=data, tempo=tempo, type='bars')

            brick = self.x_round(self.get_brick_size(data_res))

            renko_obj = pyrenko.renko()
            renko_obj.set_brick_size(brick_size = brick)
            renko_obj.build_history(prices = data.close.values, dates=data.index)
            results = pd.DataFrame(renko_obj.evaluate(), index=[brick])
            partial = {'tempo': tempo, 'brick':brick, 'balance':results.balance.iloc[0], 'sign_changes':results.sign_changes.iloc[0],
                        'price_ratio':round(results.price_ratio.iloc[0],2), 'score':round(results.score.iloc[0],2)}
            total_results.append(partial)
        
        # Select the best brick size
        final_results = pd.DataFrame(total_results).set_index('tempo')
        for i in range(len(final_results)):
            brick = final_results.sort_values('score', ascending=False).iloc[i].brick
            if brick*stop*self.leverage*contracts <= 6000*0.05: break
        score = final_results[final_results.brick == brick].score[0]

        return brick, score

    def entry_img_renko(self, renko_prices, brick, directions, renko_dates, action, price_i, time_i, stop, target):
        if not path.exists('%s_entry_images_renko' % self.symbol):
            mkdir('%s_entry_images_renko' % self.symbol)
        _, ax = plt.subplots(1, figsize=(12, 8))
        ax.set_title('%s at %s in %s'%(action, time_i, self.symbol))
        
        ax.set_xlim(0.0, len(renko_prices) + 1.0)
        ax.set_ylim(np.min(renko_prices) - brick, np.max(renko_prices) + 3*brick)
        
        for i in range(1, len(renko_prices)):
            col = 'g' if directions[i] == 1 else 'r'
            x = i
            y = renko_prices[i] - brick if directions[i] == 1 else renko_prices[i]
            ax.add_patch(patches.Rectangle((x, y), 0.8, brick, facecolor = col))
        
        num = round(len(renko_prices)/10) if len(renko_prices) > 10 else len(renko_prices)
        indexes = list(range(0,len(renko_prices), num))
        dates_list = [renko_dates[i] for i in indexes]
        plt.xticks(indexes, dates_list, rotation=45)
        
        c_in = 'darkgreen' if action == 'BUY' else 'darkred'
        dir_in = '^' if action == 'BUY' else 'v'
        bar_in = pd.Index(renko_dates).get_loc(pd.to_datetime(time_i), method='pad')
        
        plt.scatter(bar_in, price_i, c=c_in, s=150, marker=dir_in, label='entry price: %.2f'%price_i)
        plt.axhline(y=stop, color = 'red', linestyle='--', label='stop price: %.2f'%stop)
        plt.axhline(y=target, color = 'green', linestyle='--', label='target price: %.2f'%target)
        plt.axhline(y=price_i, color = 'grey', linestyle='--')
        plt.ylim((min(min(renko_prices), stop, target)-brick, max(max(renko_prices), stop, target)+brick))
        plt.legend()
        plt.savefig('%s_entry_images_renko/%s at %s(%.2f sl %.2f tp %.2f) in %s.png'%(self.symbol, action, time_i.replace(':','.'), price_i, stop, target, self.symbol))
    
    def trade_img_renko(self, renko_prices, brick, directions, renko_dates, action, price_i, price_o, time_i, time_o):
        if not path.exists('%s_trades_images_renko' % self.symbol):
            mkdir('%s_trades_images_renko' % self.symbol)
        _, ax = plt.subplots(1, figsize=(12, 8))
        ax.set_title('%s at %s-%s in %s'%(action, time_i, time_o, self.symbol))
        
        ax.set_xlim(0.0, len(renko_prices) + 1.0)
        ax.set_ylim(np.min(renko_prices) - 0.5*brick, np.max(renko_prices) + 0.5*brick)
        
        for i in range(1, len(renko_prices)):
            col = 'g' if directions[i] == 1 else 'r'
            x = i
            y = renko_prices[i] - brick if directions[i] == 1 else renko_prices[i]
            ax.add_patch(patches.Rectangle((x, y), 0.8, brick, facecolor = col))
        
        num = round(len(renko_prices)/10) if len(renko_prices) > 10 else len(renko_prices)
        indexes = list(range(0,len(renko_prices), num))
        dates_list = [renko_dates[i] for i in indexes]
        plt.xticks(indexes, dates_list, rotation=45)
        
        c_in = 'darkgreen' if action == 'BUY' else 'darkred'
        c_out = 'darkred' if action == 'BUY' else 'darkgreen'
        dir_in = '^' if action == 'BUY' else 'v'
        dir_out = 'v' if action == 'BUY' else '^'
        bar_in = pd.Index(renko_dates).get_loc(pd.to_datetime(time_i), method='pad')
        bar_out = pd.Index(renko_dates).get_loc(pd.to_datetime(time_o), method='pad')
        
        plt.scatter(bar_in, price_i, c=c_in, s=150, marker=dir_in, label='entry price: %.2f'%price_i)
        plt.scatter(bar_out, price_o, c=c_out, s=150, marker=dir_out, label='exit price: %.2f'%price_o)
        plt.legend()
        plt.savefig('%s_trades_images_renko/%s at %s(%.2f) %s(%.2f) in %s.png'%(self.symbol, action, time_i.replace(':','.'), price_i, time_o.replace(':','.'), price_o, self.symbol))
    
    def row_count(self):
        count = 0
        for _ in open('/datos/kafka2/ib_data/%s_ib_data.csv'%self.symbol): count += 1
        return count
    
    def run_strategy(self, contracts, stop, target_1, target_2, num_bars, trailing, num_days=4):
        self.print('%s %s | Nene Breakout Bot Turned On' % (self.date, self.hour))
        self.print('%s %s | Running with stop: %d & target_1: %d & target_2: %d & num_bars: %d & trailing: %d & num_days: %d'%
                (self.date, self.hour, stop, target_1, target_2, num_bars, trailing, num_days))
        # Check if operable schedule
        self.operable_schedule()
        
        if self.operable:
            # Defining Variables
            current_len = 0; last_len = 0
            new_bar = False; send_mess = False
            sent = False; exit_1 = False
            adx_calc = False
            temporalities = ['1Min', '2Min', '3Min', '4Min', '5Min', '6Min', '10Min', '12Min', '15Min', '20Min', '30Min']
            self.save_position()
            self.global_position = self.check_global_position()
            pattern_short = np.array([1., 0.])
            pattern_long = np.array([-1., 0.])
            pattern_len = len(pattern_short)
            kc_period = 20
            last_ib = 0
            init_len = 0
            calc_brick = False    ############################

            # Getting Historical Data to get renko bars
            self.print('Downloading Historical %s Data...'%self.symbol)
            self.data = self.get_historical_data()
            print(self.data.tail())
            self.print('Historical Data retrieved! from %s to %s'%(self.data.index[0], self.data.index[-1]))

            # Defining data to calculate renko according to days (input)   ###############################
            dates = np.unique(self.data.index.date)
            #now = self.data.index[-1]
            init = '%s 00:00:00'% (dates[-num_days-1])
            final = '%s 23:59:59'% dates[-2]
            #init = '%s %s' % (dates[-num_days-1], now.time())
            data_renko = self.data[init:final]
            #data_renko = self.data.loc[init:]

            # Calculating brick size
            self.print('Calculating Brick Size...'); print('Data from %s until %s'%(data_renko.index[0], data_renko.index[-1]))
            brick, score = self.optimal_brick(data_renko, stop, contracts)            # Get brick size and score
            if brick > 25: brick = 25
            #brick = 0.2 #######################################
            self.print('Brick Size: %.2f'%brick); self.print('score: %s' % score)
            if score <= 0: self.print('Score is not operable')

            # Getting Renko Bars
            self.print('Calculating Renko Bars...')
            renko_object = pyrenko.renko()                    # Defining renko object
            renko_object.set_brick_size(brick_size=brick)     # Defining brick size
            renko_object.build_history(prices=self.data.close.values,dates=self.data.index)
            prices = renko_object.get_renko_prices()
            init_len = len(prices)
            self.print(len(prices))
            self.print('Renko Bars calculated!')

            # ADX and Keltner Channel Calculation
            ## Calculate temporality to calculate ADX and Keltner Channel
            tempo_list = []
            print('Calculating ADX and Keltner Indicators...')
            for tempo in temporalities:
                data_ = self.resampler(data_renko, tempo, 'bars')
                range_ = (data_.high - data_.low).values.mean()
                partial = {'tempo': tempo, 'range': range_}
                tempo_list.append(partial)
            tempo_list_df = pd.DataFrame(tempo_list).set_index('tempo')
            tempo_list_df['diff'] = abs(tempo_list_df.range - brick)
            data_tempo = tempo_list_df.sort_values('diff', ascending=True).index[0]
            print('data tempo: %s' % data_tempo)
            ## Resampling data fo the choosen temporality
            data_ind = self.resampler(data_renko, data_tempo, 'bars')
            ## Indicators
            _, plus_di, minus_di = self.ADX(data_ind.iloc[-300:])
            midline, upper_band, lower_band = self.keltner_channel(data=data_ind.iloc[-300:], period=kc_period, mult=2)
            print('ADX and Keltner Channel Indicators Calculated!')

            #while self.weekday in [0, 1, 2, 3, 4, 6] and not (self.weekday == 4 and pd.to_datetime(self.hour).time() > pd.to_datetime('17:10:00').time()):
            for kafka_bar in self.consumer:
                # Concatening last bar kafka method
                try:
                    kafka_bar_df = pd.DataFrame(eval(str(kafka_bar.value, encoding='utf-8')), index=[0]).set_index('time')
                    kafka_bar_df.index = pd.to_datetime(kafka_bar_df.index)
                    self.data = pd.concat([self.data, kafka_bar_df])
                except: pass
                #print(self.data.tail())
                #print(self.data.iloc[[-1]])

                # Concatening last bar csv method
                '''current_ib = self.row_count()
                if current_ib != last_ib:
                    last_ib = self.row_count()
                    last_bar = self.get_last_bar()
                    if self.data.index[-1] != last_bar.index[-1]:
                        self.data = pd.concat([self.data, last_bar])
                        #print(self.data.tail())
                        print(self.data.iloc[[-1]])'''

                # Concatening last bar IB method
                '''if datetime.now().second % 5 == 0:
                    last_bar = self.get_last_bar()
                    if self.data.index[-1] != last_bar.index[-1]:
                        self.data = pd.concat([self.data, last_bar])
                        #print(self.data.tail())
                        print(self.data.iloc[[-1]])'''
                
                self.current_date()
                self.ib.sleep(1)
                #if len(self.bars) > 0 and self.bars[-1].close > 0:
                # Send confirmation message each hour to telegram
                self.continuous_check_message('%s %s | %s %s is running OK. Last Price: %.2f' % 
                                            (self.date, self.hour, self.bot_name, self.symbol, self.x_round(self.data.iloc[-1].close)))
                self.daily_results_positions()         # Send daily profit message to telegram
                self.weekly_metrics()                  # Send week metrics message to telegram
                #self.graph_trades()                   # Save daily trades graph
                
                if self.weekday <= 4 and datetime.now().hour == 0 and datetime.now().minute == 5 and not calc_brick:   ################
                #if self.weekday < 4 and self.hour == '18:05:00':
                    ######################
                    # Defining data to calculate renko according to num_days (input)
                    print('Calculating Renko bricks')
                    print(self.data.tail())
                    print(type(self.data.index))
                    try:
                        self.data.reset_index(inplace=True)
                        self.data.set_index(self.data.columns[0], inplace=True)
                        self.data.index.names = ['date']
                        print('=====')
                        print(self.data.tail())
                    except: pass
                    
                    # Defining data to calculate renko according to days (input)   ###############################
                    dates = np.unique(self.data.index.date)
                    #now = self.data.index[-1]
                    init = '%s 00:00:00'% (dates[-num_days-1])
                    final = '%s 23:59:59'% dates[-2]
                    #init = '%s %s' % (dates[-num_days-1], now.time())
                    data_renko = self.data[init:final]
                    #data_renko = self.data.loc[init:]

                    # Calculating brick size
                    self.print('Calculating Brick Size...'); self.print('Data from %s until %s'%(data_renko.index[0], data_renko.index[-1]))
                    brick, score = self.optimal_brick(data_renko, stop, contracts)
                    if brick > 25: brick = 25
                    #brick = 0.2   ############################################
                    self.print('Brick Size: %.2f'%brick); self.print('score: %s' % score)
                    if score <= 0: self.print('Score is not operable')

                    # Getting Renko Bars
                    self.print('Calculating Renko Bars...')
                    renko_object = pyrenko.renko()                    # Defining renko object
                    renko_object.set_brick_size(brick_size=brick)     # Defining brick size
                    renko_object.build_history(prices=self.data.close.values,dates=self.data.index)
                    prices = renko_object.get_renko_prices()
                    init_len = len(prices)
                    self.print(len(prices))
                    self.print('Renko Bars calculated!')
                    calc_brick = True         ##################################

                    # ADX Calculation
                    ## Calculate temporality to calculate ADX
                    tempo_list = []
                    print('Calculating ADX and Keltner Channel Indicators...')
                    for tempo in temporalities:
                        data_ = self.resampler(data_renko, tempo, 'bars')
                        range_ = (data_.high - data_.low).values.mean()
                        partial = {'tempo': tempo, 'range': range_}
                        tempo_list.append(partial)
                    tempo_list_df = pd.DataFrame(tempo_list).set_index('tempo')
                    tempo_list_df['diff'] = abs(tempo_list_df.range - brick)
                    data_tempo = tempo_list_df.sort_values('diff', ascending=True).index[0]
                    ## Resampling data fo the choosen temporality
                    data_ind = self.resampler(data_renko, data_tempo, 'bars')
                    ## Indicators
                    _, plus_di, minus_di = self.ADX(data_ind.iloc[-300:])
                    midline, upper_band, lower_band = self.keltner_channel(data=data_ind.iloc[-300:], period=kc_period, mult=2)
                    print('ADX and Keltner Channel Indicators Calculated!')

                if datetime.now().hour > 0: calc_brick = False  ##########################

                # Target condition
                if target_1 >= target_2*brick:
                    while target_1 >= target_2*brick: target_2 += 1
                
                # ADX and Keltner Actualization
                if datetime.now().minute % int(data_tempo[0]) == 0 and (datetime.now().second == 0 or datetime.now().second == 1) and not adx_calc:
                    adx_data = self.data.loc[str(pd.to_datetime(self.date)-timedelta(days=5)):]
                    ## Resampling data fo the choosen temporality
                    data_ind = self.resampler(adx_data, data_tempo, 'bars')
                    ## Indicators
                    _, plus_di, minus_di = self.ADX(data_ind.iloc[-300:])
                    midline, upper_band, lower_band = self.keltner_channel(data=data_ind.iloc[-300:], period=kc_period, mult=2)
                    #print(data_ind.tail())
                    print('%s %s | minus_di: %.2f plus_di: %.2f midline: %.2f upper_band: %.2f lower_band: %.2f'%(self.date, self.hour, 
                                list(minus_di)[-1], list(plus_di)[-1], list(midline)[-1], list(upper_band)[-1], list(lower_band)[-1]))
                    adx_calc = True
                else: adx_calc = False

                # Check Global Position
                try: self.global_position = self.check_global_position()
                except: self.ib.sleep(2); self.global_position = self.check_global_position()

                # Check if it's time to reconnect
                self.connected = self.ib.isConnected()
                self.reconnection()

                if self.connected:
                    # Calculate Renko Bars
                    #if len(self.bars) > 0 and self.bars[-1].close > 0: 
                    renko_object.do_next(self.x_round(self.data.iloc[-1].close), self.date)
                    prices = renko_object.get_renko_prices()
                    current_len = len(prices)

                    # Check new renko bar appearance
                    if current_len != last_len:
                        #num_new_bars = current_len - last_len
                        new_bar = True; last_len = current_len
                        movements = (np.array(prices) - prices[0]) / brick
                        if len(movements)>= pattern_len: self.print('%s %s | new bar %d %s'%(self.date, self.hour, current_len, movements[-pattern_len:] - movements[-1]))
                    else: new_bar = False
                    
                    # Check for Entry
                    if score > 0 and len(prices) > init_len:
                        if self.position == 0 and self.global_position == 0:             # if there's not opened positions
                            if new_bar and current_len >= kc_period+pattern_len-2:
                                recognition = movements[-pattern_len:] - movements[-1]   # pattern filter
                                self.print('minus_di: %.2f , plus_di: %.2f , midline: %.2f , upper_band: %.2f , lower_band: %.2f'%(list(minus_di)[-1], 
                                                    list(plus_di)[-1], list(midline)[-1],  list(upper_band)[-1], list(lower_band)[-1]))
                                
                                # Entry conditions
                                if not (self.weekday == 4 and pd.to_datetime(self.hour).time() > pd.to_datetime('12:00:00').time()):
                                    ## Sells
                                    if not sent and sum(recognition == pattern_short) == pattern_len and prices[-2] > lower_band[-2] and prices[-1] < lower_band[-1] and minus_di[-1] > 40 and minus_di[-1] > plus_di[-1]:
                                        max_stop = brick*stop*self.leverage*contracts
                                        price_sell_in_1, sl_sell_1, tp_sell_1, time_sell_in, comm_sell_in_1, profit_sell, ord_sell_sl_1, ord_sell_tp_1 = self.braket_market('SELL', contracts/2, stop*brick, target_1, max_stop)
                                        price_sell_in_2, sl_sell_2, tp_sell_2, time_sell_in, comm_sell_in_2, profit_sell, ord_sell_sl_2, ord_sell_tp_2 = self.braket_market('SELL', contracts/2, stop*brick, target_2*brick, max_stop, entry_price=price_sell_in_1)
                                        if price_sell_in_1 > 0 and price_sell_in_2 > 0: sent = True
                                        bar_entry = len(movements)
                                        #trailing_stop = trailing * brick
                                        tr_1 = self.x_round(trailing * target_1)
                                        tr_2 = self.x_round(trailing * target_2 * brick)
                                        exit_1 = False

                    # Check for Exit
                    if self.position != 0:
                        ## Sells
                        if self.position < 0:
                            ## Trailing Stop ==========
                            try:
                                if not exit_1: sl_sell_1 = self.trailing_stop(price_in=price_sell_in_1, trailing=trailing_stop, sl=sl_sell_1, order=ord_sell_sl_1)
                                sl_sell_2 = self.trailing_stop(price_in=price_sell_in_1, trailing=trailing_stop, sl=sl_sell_2, order=ord_sell_sl_2)
                            except: self.print('Trying to apply trailing stop... Order has been Filled!')

                            # By stop ==========
                            ## Stop 1
                            if self.check_pendings(ord_sell_sl_1) and not exit_1 and self.position < 0:   # Check if stop 1 is filled
                                self.exit_pending(ord_sell_sl_1, 'SELL', contracts/2, price_sell_in_1, time_sell_in, comm_sell_in_1, 'sl1')
                                exit_1 = True; sent = False
                            ## Stop 2
                            if self.check_pendings(ord_sell_sl_2) and self.position < 0:   # Check if stop 2 is filled
                                self.exit_pending(ord_sell_sl_2, 'SELL', contracts/2, price_sell_in_2, time_sell_in, comm_sell_in_2, 'sl2')
                                sent = False
                            
                            ## False Stop
                            ### Stop 1
                            if not self.check_pendings(ord_sell_sl_1) and self.data.iloc[-1].high - sl_sell_1 >= 2 and not exit_1 and self.position < 0:
                                self.exit_market(ord_sell_tp_1, 'SELL', contracts/2, price_sell_in_1, time_sell_in, comm_sell_in_1, 'fsl1')
                                exit_1 = True; sent = False
                            ### Stop 2
                            if not self.check_pendings(ord_sell_sl_2) and self.data.iloc[-1].high - sl_sell_2 >= 2 and self.position < 0:
                                self.exit_market(ord_sell_tp_2, 'SELL', contracts/2, price_sell_in_2, time_sell_in, comm_sell_in_2, 'fsl2')
                                sent = False
                                
                            # By target ==========
                            ## Target 1
                            if self.check_pendings(ord_sell_tp_1) and not exit_1 and self.position < 0:    # Check if target 1 is filled
                                self.exit_pending(ord_sell_tp_1, 'SELL', contracts/2, price_sell_in_1, time_sell_in, comm_sell_in_1, 'tp1')
                                exit_1 = True; sent = False
                            ## Target 2
                            if self.check_pendings(ord_sell_tp_2) and self.position < 0:                   # Check if target 2 is filled
                                self.exit_pending(ord_sell_tp_2, 'SELL', contracts/2, price_sell_in_2, time_sell_in, comm_sell_in_2, 'tp2')
                                sent = False

                            # by number of bars
                            if len(movements) - bar_entry >= num_bars and self.position < 0:              # Number of bars exit condition
                                if self.position == -contracts:
                                    self.exit_market(ord_sell_tp_1, 'SELL', contracts/2, price_sell_in_1, time_sell_in, comm_sell_in_1, 'bars')
                                    self.exit_market(ord_sell_tp_2, 'SELL', contracts/2, price_sell_in_2, time_sell_in, comm_sell_in_2, 'bars')
                                    sent = False
                                else:
                                    self.exit_market(ord_sell_tp_2, 'SELL', contracts/2, price_sell_in_2, time_sell_in, comm_sell_in_2, 'bars')
                                    sent = False

                            # Exit on friday
                            if self.weekday == 4 and pd.to_datetime(self.hour).time() >= pd.to_datetime('16:57:00').time() and self.position < 0:
                                if self.position == -contracts:
                                    self.exit_market(ord_sell_tp_1, 'SELL', contracts/2, price_sell_in_1, time_sell_in, comm_sell_in_1, 'fri')
                                    self.exit_market(ord_sell_tp_2, 'SELL', contracts/2, price_sell_in_2, time_sell_in, comm_sell_in_2, 'fri')
                                    sent = False
                                else:
                                    self.exit_market(ord_sell_tp_2, 'SELL', contracts/2, price_sell_in_2, time_sell_in, comm_sell_in_2, 'fri')
                                    sent = False

                else:
                    if not self.interrumption:
                        try:
                            self.print('Trying to reconnect...')
                            self.ib.disconnect()
                            self.ib.sleep(10)
                            self.ib.connect('127.0.0.1', self.port, self.client)
                            self.connected = self.ib.isConnected()
                            if self.connected:
                                self.print('Connection reestablished!')
                                self.print('Getting Data...')
                                self.data = self.get_historical_data()
                                renko_object.build_history(prices=self.data.close.values, dates=self.data.index)
                                prices = renko_object.get_renko_prices()
                                #directions = renko_object.get_renko_directions()
                                #renko_dates = renko_object.get_renko_dates()
                                #self.bars = self.ib.reqRealTimeBars(self.contract, 5, 'TRADES', False)
                                #self.print('Last Close of %s: %.2f' % (self.symbol, self.bars[-1].close))
                                #self.print('%s Data has been Updated!' % self.symbol)
                        except:
                            self.print('Connection Failed! Trying to reconnect in 10 seconds...')

            self.ib.disconnect()
            self.print('%s %s | Session Ended. Good Bye!' % (self.date, self.hour))

if __name__ == '__main__':
    symbol = 'MNQ'
    port = 7497
    client = 78

    live_nene = LiveNene(symbol=symbol, bot_name='Nene Breakout Shorts (demo)', temp='1 min', port=port, client=client, real=False)
    live_nene.run_strategy(contracts=2, stop=1, target_1=7, target_2=7, num_bars=10, trailing=0.8, num_days=3)