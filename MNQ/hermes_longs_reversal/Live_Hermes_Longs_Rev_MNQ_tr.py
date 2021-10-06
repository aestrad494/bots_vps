#
# Python Script with live trading 
# Hermes Strategy Class
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

class LiveHermes(Live, Indicators):
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
    
    def LR_val(self, values, period):
        '''Get LR value for the last bar

        Parameters:
            values (list): values (prices) to calculate Linear Regression value
            period (int): Linear Regression period

        Returns:
            float: last value Linear Regression 
        '''
        sum_per = sum(range(1,period+1))
        sum_per_w = sum([x**2 for x in range(1,period+1)])
        sum_closes = values.sum()
        sum_closes_w = sum([(i+1) * values[i] for i in range(len(values))])
        b = ((period * sum_closes_w) - (sum_per * sum_closes)) / ((period * sum_per_w) - (sum_per**2))
        a = (sum_closes - (b*sum_per))/period
        LR = a + b * period
        return LR

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
    
    def run_strategy(self, contracts, pattern, stop, target_1, target_2, num_bars, trailing, lr_period=30, lr_previous=30, lr_distance=5, num_days=4, pause_time=4):
        self.print('%s %s | Hermes Longs Reversal 2 Contracts Bot Turned On' % (self.date, self.hour))
        self.print('%s %s | Running with stop: %d & target_1: %d & target_2: %d & num_bars: %d & trailing: %.2f & lr_period: %d & lr_previous: %d & lr_distance: %d & num_days: %d & pause_time: %d'%
                (self.date, self.hour, stop, target_1, target_2, num_bars, trailing, lr_period, lr_previous, lr_distance, num_days, pause_time))
        # Check if operable schedule
        self.operable_schedule()

        if self.operable:
            # Defining Variables
            current_len = 0; last_len = 0
            first_lr = False; new_bar = False
            sent = False; exit_1 = False
            self.save_position()
            self.global_position = self.check_global_position()
            pattern_len = len(pattern)
            exit_by_tp = False; reversal = False
            allow_hour = 0; num_bars_rev = 0
            last_ib = 0
            init_len = 0
            calc_brick = False      ##############################

            #target condition
            if target_1 == target_2: target_2 += 1

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
            self.print('Calculating Brick Size...'); self.print('Data from %s until %s'%(data_renko.index[0], data_renko.index[-1]))
            brick, score = self.optimal_brick(data_renko, stop, contracts)
            if brick > 25: brick = 25
            #brick = 1 ##########################################
            self.print('Brick Size: %.2f'%brick); self.print('score: %s' % score)
            if score <= 0: self.print('Score is not operable')

            # Getting Renko Bars
            self.print('Calculating Renko Bars...')
            renko_object = pyrenko.renko()                    # Defining renko object
            renko_object.set_brick_size(brick_size=brick)     # Defining brick size
            renko_object.build_history(prices=self.data.close.values,dates=self.data.index)
            prices = renko_object.get_renko_prices()
            init_len = len(prices)
            #directions = renko_object.get_renko_directions()
            #renko_dates = renko_object.get_renko_dates()
            self.print(len(prices))
            self.print('Renko Bars calculated!')

            #while self.weekday in [0, 1, 2, 3, 4, 6] and not (self.weekday == 4 and pd.to_datetime(self.hour).time() > pd.to_datetime('17:10:00').time()):
            for kafka_bar in self.consumer:    
                
                # Concatening last bar kafka method
                try:
                    kafka_bar_df = pd.DataFrame(eval(str(kafka_bar.value, encoding='utf-8')), index=[0]).set_index('time')
                    kafka_bar_df.index = pd.to_datetime(kafka_bar_df.index)
                    self.data = pd.concat([self.data, kafka_bar_df])
                except: pass

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
                self.continuous_check_message('%s %s | %s %s is running OK. Last price: %.2f' % 
                                                    (self.date, self.hour, self.bot_name, self.symbol, self.x_round(self.data.iloc[-1].close)))
                self.daily_results_positions()         # Send daily profit message to telegram
                self.weekly_metrics()                  # Send week metrics message to telegram
                #self.graph_trades()                   # Save daily trades graph

                # Allow or not trading according to target exit
                if exit_by_tp:
                    allow_entry = False
                    if datetime.now() - allow_hour >= timedelta(hours=0):
                        exit_by_tp = False; reversal = False
                        self.print('%s %s | Entry trades allowed again!' % (self.date, self.hour))
                else: allow_entry = True

                if self.weekday <= 4 and datetime.now().hour == 0 and datetime.now().minute == 5 and not calc_brick:   ################
                #if self.weekday < 4 and self.hour == '18:05:00':
                    ######################
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
                    #brick = 1 ###################################################
                    self.print('Brick Size: %.2f'%brick); self.print('score: %s' % score)
                    if score <= 0: self.print('Score is not operable')

                    # Getting Renko Bars
                    self.print('Calculating Renko Bars...')
                    renko_object = pyrenko.renko()                    # Defining renko object
                    renko_object.set_brick_size(brick_size=brick)     # Defining brick size
                    renko_object.build_history(prices=self.data.close.values,dates=self.data.index)
                    prices = renko_object.get_renko_prices()
                    init_len = len(prices)
                    #directions = renko_object.get_renko_directions()
                    #renko_dates = renko_object.get_renko_dates()
                    self.print(len(prices))
                    self.print('Renko Bars calculated!')
                    calc_brick = True         ##################################
                
                if datetime.now().hour > 0: calc_brick = False  ##########################

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
                    #directions = renko_object.get_renko_directions()
                    #renko_dates = renko_object.get_renko_dates()
                    current_len = len(prices)

                    # Check new renko bar appearance
                    if current_len != last_len:
                        num_new_bars = current_len - last_len
                        new_bar = True; last_len = current_len
                        movements = (np.array(prices) - prices[0]) / brick
                        if len(movements)>= pattern_len: self.print('%s %s | new bar %d %s'%(self.date, self.hour, current_len, movements[-pattern_len:] - movements[-1]))
                    else: new_bar = False

                    if current_len >= lr_previous+lr_period+pattern_len-2:
                        if new_bar:
                            # Calculate new lr movement value
                            if not first_lr:
                                lr_movements = self.LR(data = pd.Series(movements), period=lr_period).values
                                first_lr = True
                            else:
                                for j in range(-num_new_bars, 0 ,1):
                                    lr_movements = np.append(lr_movements, self.LR_val(movements[len(movements)-lr_period+(j+1):len(movements)+(j+1)], lr_period))   ####'''
                    
                    # Check for Entry
                    #if score != 0.001:
                    if score > 0 and len(prices) > init_len:
                        if self.position == 0 and self.global_position == 0:             # if there's not opened positions
                            if new_bar and current_len >= lr_previous+lr_period+pattern_len-2:
                                recognition = movements[-pattern_len:] - movements[-1]   # pattern filter
                                lr_i = lr_movements[-pattern_len]                        # LR filter
                                lr_prev = lr_movements[-pattern_len-lr_previous+1]       # LR previous filter
                                self.print('%s %s'%(round(lr_i - lr_prev, 2), lr_distance))

                                # Entry conditions
                                if not (self.weekday == 4 and pd.to_datetime(self.hour).time() > pd.to_datetime('12:00:00').time()):
                                    ## Buys
                                    if not sent and sum(recognition == pattern) == pattern_len and lr_i - lr_prev >= lr_distance and allow_entry:
                                    #if not sent and sum(recognition == pattern) != pattern_len and lr_i - lr_prev != 0 and allow_entry:
                                        max_stop = brick*stop*self.leverage*contracts
                                        price_buy_in_1, sl_buy_1, tp_buy_1, time_buy_in, comm_buy_in_1, profit_buy, ord_buy_sl_1, ord_buy_tp_1 = self.braket_market('BUY', contracts/2, stop*brick, target_1*brick, max_stop)
                                        price_buy_in_2, sl_buy_2, tp_buy_2, time_buy_in, comm_buy_in_2, profit_buy, ord_buy_sl_2, ord_buy_tp_2 = self.braket_market('BUY', contracts/2, stop*brick, target_2*brick, max_stop, entry_price=price_buy_in_1)
                                        if price_buy_in_1 > 0 and price_buy_in_2 > 0: sent = True
                                        bar_entry = len(movements)
                                        #trailing_stop = trailing * brick
                                        tr_1 = self.x_round(trailing * target_1 * brick)
                                        tr_2 = self.x_round(trailing * target_2 * brick)
                                        exit_1 = False
                                        reversal = False
                                    
                                    ## Sells
                                    if not sent and sum(recognition == pattern) == pattern_len and lr_i - lr_prev != 0 and reversal and len(prices) - num_bars_rev <= 1:
                                        max_stop = brick*stop*self.leverage*contracts
                                        price_sell_in_1, sl_sell_1, tp_sell_1, time_sell_in, comm_sell_in_1, profit_sell, ord_sell_sl_1, ord_sell_tp_1 = self.braket_market('SELL', contracts/2, stop*brick, target_1*brick, max_stop)
                                        price_sell_in_2, sl_sell_2, tp_sell_2, time_sell_in, comm_sell_in_2, profit_sell, ord_sell_sl_2, ord_sell_tp_2 = self.braket_market('SELL', contracts/2, stop*brick, target_2*brick, max_stop, entry_price=price_sell_in_1)
                                        if price_sell_in_1 > 0 and price_sell_in_2 > 0: sent = True
                                        bar_entry = len(movements)
                                        #trailing_stop = trailing * brick
                                        tr_1 = self.x_round(trailing * target_1 * brick)
                                        tr_2 = self.x_round(trailing * target_2 * brick)
                                        exit_1 = False
                                    
                    # Check for Exit
                    if self.position != 0:
                        ## Buys
                        if self.position > 0:
                            # trailing stop ==========
                            #if not exit_1: sl_buy_1 = self.trailing_stop(price_in=price_buy_in_1, trailing=trailing_stop, sl=sl_buy_1, order=ord_buy_sl_1)
                            #sl_buy_2 = self.trailing_stop(price_in=price_buy_in_1, trailing=trailing_stop, sl=sl_buy_2, order=ord_buy_sl_2)
                            try:
                                if not exit_1: sl_buy_1 = self.trailing_stop(price_in=price_buy_in_1, trailing=tr_1, sl=sl_buy_1, order=ord_buy_sl_1)
                                sl_buy_2 = self.trailing_stop(price_in=price_buy_in_1, trailing=tr_2, sl=sl_buy_2, order=ord_buy_sl_2)
                            except: self.print('Trying to apply trailing stop... Order has been Filled!')

                            # By stop ==========
                            ## Stop 1
                            if self.check_pendings(ord_buy_sl_1) and not exit_1 and self.position > 0:   # Check if stop 1 is filled
                                self.exit_pending(ord_buy_sl_1, 'BUY', contracts/2, price_buy_in_1, time_buy_in, comm_buy_in_1, 'sl1')
                                exit_1 = True; sent = False 
                            ## Stop 2
                            if self.check_pendings(ord_buy_sl_2) and self.position > 0:   # Check if stop 2 is filled
                                self.exit_pending(ord_buy_sl_2, 'BUY', contracts/2, price_buy_in_2, time_buy_in, comm_buy_in_2, 'sl2')
                                sent = False
                            
                            ## False Stop
                            ### Stop 1
                            if not self.check_pendings(ord_buy_sl_1) and sl_buy_1 - self.data.iloc[-1].low >= 2 and not exit_1 and self.position > 0:
                                self.exit_market(ord_buy_tp_1, 'BUY', contracts/2, price_buy_in_1, time_buy_in, comm_buy_in_1, 'fsl1')
                                exit_1 = True; sent = False
                            ### Stop 2
                            if not self.check_pendings(ord_buy_sl_2) and sl_buy_2 - self.data.iloc[-1].low >= 2 and self.position > 0:
                                self.exit_market(ord_buy_tp_2, 'BUY', contracts/2, price_buy_in_2, time_buy_in, comm_buy_in_2, 'fsl2')
                                sent = False
                            
                            # By target ==========
                            ## Target 1
                            if self.check_pendings(ord_buy_tp_1) and not exit_1 and self.position > 0:    # Check if target 1 is filled
                                self.exit_pending(ord_buy_tp_1, 'BUY', contracts/2, price_buy_in_1, time_buy_in, comm_buy_in_1, 'tp1')
                                exit_1 = True; sent = False
                            ## Target 2
                            if self.check_pendings(ord_buy_tp_2) and self.position > 0:                   # Check if target 2 is filled
                                self.exit_pending(ord_buy_tp_2, 'BUY', contracts/2, price_buy_in_2, time_buy_in, comm_buy_in_2, 'tp2')
                                sent = False
                                allow_hour = datetime.now() + timedelta(hours=pause_time)
                                self.print('%s %s | Pause Activation...(%s)' % (self.date, self.hour, allow_hour))
                                num_bars_rev = current_len; self.print(num_bars_rev)
                                reversal = True
                                exit_by_tp = True
                                self.ib.sleep(30)
                                
                            # by number of bars
                            if len(movements) - bar_entry >= num_bars and self.position > 0:              # Number of bars exit condition
                                if self.position == contracts:
                                    self.exit_market(ord_buy_tp_1, 'BUY', contracts/2, price_buy_in_1, time_buy_in, comm_buy_in_1, 'bars')
                                    self.exit_market(ord_buy_tp_2, 'BUY', contracts/2, price_buy_in_2, time_buy_in, comm_buy_in_2, 'bars')
                                    sent = False
                                else:
                                    self.exit_market(ord_buy_tp_2, 'BUY', contracts/2, price_buy_in_2, time_buy_in, comm_buy_in_2, 'bars')
                                    sent = False

                            # Exit on friday
                            if self.weekday == 4 and pd.to_datetime(self.hour).time() >= pd.to_datetime('16:57:00').time() and self.position > 0:
                                if self.position == contracts:
                                    self.exit_market(ord_buy_tp_1, 'BUY', contracts/2, price_buy_in_1, time_buy_in, comm_buy_in_1, 'fri')
                                    self.exit_market(ord_buy_tp_2, 'BUY', contracts/2, price_buy_in_2, time_buy_in, comm_buy_in_2, 'fri')
                                    sent = False
                                else:
                                    self.exit_market(ord_buy_tp_2, 'BUY', contracts/2, price_buy_in_2, time_buy_in, comm_buy_in_2, 'fri')
                                    sent = False
                        
                        ## Sells
                        if self.position < 0:
                            ## Trailing Stop ==========
                            #if not exit_1: sl_sell_1 = self.trailing_stop(price_in=price_sell_in_1, trailing=trailing_stop, sl=sl_sell_1, order=ord_sell_sl_1)
                            #sl_sell_2 = self.trailing_stop(price_in=price_sell_in_1, trailing=trailing_stop, sl=sl_sell_2, order=ord_sell_sl_2)
                            try:
                                if not exit_1: sl_sell_1 = self.trailing_stop(price_in=price_sell_in_1, trailing=tr_1, sl=sl_sell_1, order=ord_sell_sl_1)
                                sl_sell_2 = self.trailing_stop(price_in=price_sell_in_1, trailing=tr_2, sl=sl_sell_2, order=ord_sell_sl_2)
                            except: self.print('Trying to apply trailing stop... Order has been Filled!')

                            # By stop ==========
                            ## Stop 1
                            if self.check_pendings(ord_sell_sl_1) and not exit_1 and self.position < 0:   # Check if stop 1 is filled
                                self.exit_pending(ord_sell_sl_1, 'SELL', contracts/2, price_sell_in_1, time_sell_in, comm_sell_in_1, 'sl1')
                                exit_1 = True; sent = False; reversal = False
                            ## Stop 2
                            if self.check_pendings(ord_sell_sl_2) and self.position < 0:   # Check if stop 2 is filled
                                self.exit_pending(ord_sell_sl_2, 'SELL', contracts/2, price_sell_in_2, time_sell_in, comm_sell_in_2, 'sl2')
                                sent = False; reversal = False
                            
                            ## False Stop
                            #if 6 <= datetime.now().hour <= 20:
                            ### Stop 1
                            if not self.check_pendings(ord_sell_sl_1) and self.data.iloc[-1].high - sl_sell_1 >= 2 and not exit_1 and self.position < 0:
                                self.exit_market(ord_sell_tp_1, 'SELL', contracts/2, price_sell_in_1, time_sell_in, comm_sell_in_1, 'fsl1')
                                exit_1 = True; sent = False; reversal = False
                            ### Stop 2
                            if not self.check_pendings(ord_sell_sl_2) and self.data.iloc[-1].high - sl_sell_2 >= 2 and self.position < 0:
                                self.exit_market(ord_sell_tp_2, 'SELL', contracts/2, price_sell_in_2, time_sell_in, comm_sell_in_2, 'fsl2')
                                sent = False; reversal = False
                            
                            # By target ==========
                            ## Target 1
                            if self.check_pendings(ord_sell_tp_1) and not exit_1 and self.position < 0:    # Check if target 1 is filled
                                self.exit_pending(ord_sell_tp_1, 'SELL', contracts/2, price_sell_in_1, time_sell_in, comm_sell_in_1, 'tp1')
                                exit_1 = True; sent = False; reversal = False
                            ## Target 2
                            if self.check_pendings(ord_sell_tp_2) and self.position < 0:                   # Check if target 2 is filled
                                self.exit_pending(ord_sell_tp_2, 'SELL', contracts/2, price_sell_in_2, time_sell_in, comm_sell_in_2, 'tp2')
                                sent = False; reversal = False

                            # by number of bars
                            if len(movements) - bar_entry >= num_bars and self.position < 0:              # Number of bars exit condition
                                if self.position == -contracts:
                                    self.exit_market(ord_sell_tp_1, 'SELL', contracts/2, price_sell_in_1, time_sell_in, comm_sell_in_1, 'bars')
                                    self.exit_market(ord_sell_tp_2, 'SELL', contracts/2, price_sell_in_2, time_sell_in, comm_sell_in_2, 'bars')
                                    sent = False; reversal = False
                                else:
                                    self.exit_market(ord_sell_tp_2, 'SELL', contracts/2, price_sell_in_2, time_sell_in, comm_sell_in_2, 'bars')
                                    sent = False; reversal = False

                            # Exit on friday
                            if self.weekday == 4 and pd.to_datetime(self.hour).time() >= pd.to_datetime('16:57:00').time() and self.position < 0:
                                if self.position == -contracts:
                                    self.exit_market(ord_sell_tp_1, 'SELL', contracts/2, price_sell_in_1, time_sell_in, comm_sell_in_1, 'fri')
                                    self.exit_market(ord_sell_tp_2, 'SELL', contracts/2, price_sell_in_2, time_sell_in, comm_sell_in_2, 'fri')
                                    sent = False; reversal = False
                                else:
                                    self.exit_market(ord_sell_tp_2, 'SELL', contracts/2, price_sell_in_2, time_sell_in, comm_sell_in_2, 'fri')
                                    sent = False; reversal = False

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
    client = 60
    pattern = np.array([-5., -4., -3., -2., -1., 0.])
    
    live_hermes = LiveHermes(symbol=symbol, bot_name='Hermes Longs Rev (demo)', temp='1 min', port=port, client=client, real=False)
    live_hermes.run_strategy(contracts=2, pattern=pattern, stop=3, target_1=4, target_2=6, num_bars=7, trailing=0.8, lr_period=20, 
            lr_previous=10, lr_distance=2, num_days=3, pause_time=2)