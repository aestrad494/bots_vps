#
# Python Script with live trading Class
#
# (c) Andres Estrada Cadavid
# QSociety

import pandas as pd
from datetime import datetime, timedelta
import requests
from ib_insync import IB, Future, Forex, Stock, util
#from platform import system
from os import path, mkdir
from arctic import Arctic
from arctic.date import DateRange
from arctic.hooks import register_get_auth_hook
from arctic.auth import Credential
#from tzlocal import get_localzone
import numpy as np
import json

class Generator():
    def __init__(self, symbol, bot_name, port, client):
        self.symbol = symbol
        self.bot_name = bot_name
        instruments = pd.read_csv('instruments.csv').set_index('symbol')
        self.parameters = instruments.loc[self.symbol]
        self.market = str(self.parameters.market)
        self.exchange = str(self.parameters.exchange)
        self.tick_size = float(self.parameters.tick_size)
        #self.digits = int(self.parameters.digits)
        #self.leverage = int(self.parameters.leverage)
        self.port = port
        self.client = client
        self.current_date()
        self.initialize_csv()
        self.print('Trying to Connect to Trading Platform...', 'w')
        self._sundays_activation()
        self.ib = IB()
        self.print(self.ib.connect('127.0.0.1', port, self.client))
        self.connected = self.ib.isConnected()
        self._get_contract()
        self.interrumption = False
        self.bars = self.ib.reqRealTimeBars(self.contract, 5, 'TRADES', False)
        # Variables
        self.cont_mess = False
        self.current_len = 0
        self.last_len = 0
        #self.position = 0
        self.operable = True
        #self.local_id = 0
    
    def print(self, message, type='a'):
        sample = open('logs/%s_logs.txt'%self.symbol, type)
        print(message, file=sample)
        print(message)
        sample.close()

    def x_round(self, x):
        '''Round price values according to tick size

        Parameters:
            x (float): number to be rounded
        
        Returns:
            float: rounded number
        '''
        mult = 1/self.tick_size
        return round(x*mult)/mult
    
    def json_serializer(self, data):
        return json.dumps(data).encode("utf-8")

    def initialize_csv(self):
        '''Create CSV files when they don't exist
        '''
        if not path.exists('logs'):
            mkdir('logs')
        if not path.exists('ib_data'):
            mkdir('ib_data')
        if not path.exists('historical_data'):
            mkdir('historical_data')
        if not path.exists('last_bar'):
            mkdir('last_bar')
        
    def current_date(self):
        '''Get current date, weekday and hour
        '''
        self.date = datetime.now().strftime('%Y-%m-%d')
        self.weekday = datetime.now().weekday()
        self.hour = datetime.now().strftime('%H:%M:%S')
    
    '''def continuous_check_message(self, message):
        ''Every hour message confirmation

        Parameters:
            message (str): Message to be sent
        ''
        if datetime.now().minute == 0 and datetime.now().second == 0:
            self.send_telegram_message(message)
            if not datetime.now().hour == 18:
                self.check_data_farm()'''
    
    def continuous_check_message(self, message):
        '''Every hour message confirmation

        Parameters:
            message (str): Message to be sent
        '''
        if datetime.now().minute == 0 and not self.cont_mess:
            self.send_telegram_message(message)
            self.cont_mess = True
            if not datetime.now().hour == 18:
                self.check_data_farm()
        if datetime.now().minute != 0:
            self.cont_mess = False

    def check_data_farm(self):
        '''Send telegram message if Streaming Data is paused
        '''
        self.current_len = len(self.bars)
        if self.current_len != self.last_len:
            self.last_len = self.current_len
        else:
            try:
                self.ib.disconnect()
                self.ib.connect('127.0.0.1', self.port, self.client)
                if datetime.now().hour != 18:
                    while self.current_len == self.last_len:
                        self.current_len = len(self.bars)
                        self.ib.reqRealTimeBars(self.contract, 5, 'TRADES', False)
                        self.ib.sleep(5)
                        self.ib.cancelRealTimeBars(self.bars)
                        if datetime.now().second() == 0:
                            message = 'Streaming Data is paused. Check IB Data Farm!\n'
                            message += 'Last Price = %.2f' % self.bars[-1].close
                            self.send_telegram_message(message, type='alarm')
                    if self.current_len != self.last_len:
                        message = 'Streaming Data is OK!'
                        self.send_telegram_message(message, type='alarm')
                        self.last_len = self.current_len
            except: pass

    def _sundays_activation(self):
        '''Sundays bot Activation when market opens
        '''
        hour = '18:00:05'
        if self.weekday == 6:
            if pd.to_datetime(self.hour).time() < pd.to_datetime(hour).time():
                self.print('Today is Sunday. Bot activation is at 18:00:00')
                while True:
                    self.current_date()
                    if pd.to_datetime(self.hour).time() >= pd.to_datetime(hour).time():
                        self.print('Activation Done')
                        message = '%s %s | Bot Activation Done. %s %s'%(self.date, self.hour, self.bot_name, self.symbol)
                        self.send_telegram_message(message)
                        break
    
    def operable_schedule(self):
        '''Defines operable schedules
        '''
        if self.weekday == 4 and pd.to_datetime(self.hour).time() > pd.to_datetime('18:00:00').time():
            self.print('%s %s | Today is Friday and Market has Closed!'%(self.date, self.hour))
            self.operable = False
        elif self.weekday == 5:
            self.print('%s %s | Today is Saturday and market is not Opened'%(self.date, self.hour))
            self.operable = False
        else:
            self.operable = True

    def _local_symbol_selection(self):
        '''Selects local symbol according to symbol and current date

        Returns:
            str:local symbol according to symbol and current date
        '''
        current_date = datetime.now().date()
        if self.symbol in ['ES', 'RTY', 'NQ', 'MES', 'MNQ', 'M2K']:
            contract_dates = pd.read_csv('contract_dates/indexes_globex.txt', parse_dates=True)
        elif self.symbol in ['YM', 'MYM', 'DAX']:
            contract_dates = pd.read_csv('contract_dates/indexes_ecbot_dtb.txt', parse_dates=True)
        elif self.symbol in ['QO', 'MGC']:
            contract_dates = pd.read_csv('contract_dates/QO_MGC.txt', parse_dates=True)
        elif self.symbol in ['CL', 'QM']: 
            contract_dates = pd.read_csv('contract_dates/CL_QM.txt', parse_dates=True)
        else: 
            contract_dates = pd.read_csv('contract_dates/%s.txt'%self.symbol, parse_dates=True)

        # Current contract selection according to current date
        for i in range(len(contract_dates)):
            initial_date = pd.to_datetime(contract_dates.iloc[i].initial_date).date()
            final_date = pd.to_datetime(contract_dates.iloc[i].final_date).date()
            if initial_date <= current_date <= final_date:
                current_contract = contract_dates.iloc[i].contract
                break
        
        # local symbol selection
        local = current_contract
        if self.symbol in ['ES', 'RTY', 'NQ', 'MES', 'MNQ', 'M2K', 'QO', 'CL', 'MGC', 'QM']:
            local = '%s%s'%(self.symbol, current_contract)
        if self.symbol in ['YM', 'ZS']: local = '%s   %s'%(self.symbol, current_contract)
        if self.symbol == 'MYM': local = '%s  %s'%(self.symbol, current_contract)
        if self.symbol == 'DAX': local = 'FDAX %s'%current_contract
        
        return local

    def _get_contract(self):
        '''Get current contract given symbol and current date
        '''
        if self.market == 'futures':
            local = self._local_symbol_selection()
            self.contract = Future(symbol=self.symbol, exchange=self.exchange, localSymbol=local)
        elif self.market == 'forex':
            self.contract = Forex(self.symbol)
        elif self.market == 'stocks':
            self.contract = Stock(symbol=self.symbol, exchange=self.exchange, currency='USD')
    
    def arctic_auth_hook(self, mongo_host, app, database):
        '''Mongo Authentication
        '''
        credentials = pd.read_csv('mongo_credentials.csv')
        user = credentials['user'].iloc[0]
        password = credentials['password'].iloc[0]
        return Credential(database='arctic',user=user,password=password,)

    def get_arctic_data(self, start, end='', tempo='5S'):
        '''Retrieves and prepares historical data (from Arctic)

        Parameters:
            start (str): start date to download
            end (str): end date to download. if '' then download up to the latest date
            tempo (str): temporality bars
        
        Returns:
            DataFrame: historical Arctic data resampled according to tempo
        '''
        register_get_auth_hook(self.arctic_auth_hook)
        store = Arctic('db')#('157.245.223.103')#('db')
        library = store['Futures_Historical_Ticks']
        if end == '':
            #data = library.read(self.symbol, date_range=DateRange(start='%s 00:00:00' % start)).data
            data = library.read(self.symbol, date_range=DateRange('%s' % start)).data
        else:
            data = library.read(self.symbol, date_range=DateRange(start='%s 00:00:00' % start, end='%s 23:59:59' % end)).data
        # Resampling according to temporality
        if tempo == 'ticks':
            resampled_data = data
        else:
            resampled_data = self.resampler(data=data, tempo=tempo, type='ticks')
        
        return resampled_data

    def _download_partial_data(self, date, duration, tempo, timezone):
        '''Download snippet historical data

        Parameters:
            date (str): reference date to download
            duration (str): total time to download
            tempo (str): temporality to download data
            timezone (str): localize timezone
        
        Returns:
            DataFrame: historical data
        '''
        bars = self.ib.reqHistoricalData(self.contract, endDateTime=date, durationStr=duration, barSizeSetting=tempo,
            whatToShow='TRADES', useRTH=False, formatDate=1)
        bars_df = util.df(bars)[['date', 'open', 'high', 'low', 'close', 'volume']].set_index('date')
        bars_df.index = bars_df.index.tz_localize(timezone).tz_convert('US/Eastern').tz_localize(None)
        return bars_df

    def download_complete(self, until, timezone):
        '''Download remaining data

        Parameters:
            until (str): date reference to download data
            timezone (str): convert timezone

        Returns:
            DataFrame: Downloaded data from 'until' date until now
        '''
        #initial Download
        bars_df = []; tries = 0
        while len(bars_df) == 0:
            bars_df = self._download_partial_data('', '3600 S', '5 secs', timezone)
            tries += 1; self.ib.sleep(1)
            print('%d try'%tries, end='\r') if tries <= 1 else print('%d tries'%tries, end='\r')
        init = str(bars_df.index[0].tz_localize('US/Eastern').tz_convert(timezone).tz_localize(None)).replace('-', '')
        #loop Download
        if until < pd.to_datetime(init):
            while until < pd.to_datetime(init):
                bars_i_df = self._download_partial_data(init, '3600 S', '5 secs', timezone)
                bars_df = pd.concat([bars_i_df, bars_df])
                init = str(bars_df.index[0].tz_localize('US/Eastern').tz_convert(timezone).tz_localize(None)).replace('-', '')
        return bars_df.loc[until:].iloc[1:]
    
    def download_historical(self, days_before=14):
        '''Download Historical Data combining arctic and TWS data

        Parameters:
            days_before (int): Days before to Download
        
        Returns:
            DataFrame: Historical Data
        '''
        data_loaded = False
        start = str(pd.to_datetime(self.date).date() - timedelta(days=days_before))        # get start date to download arctic data (14 days before)
        
        #data = self.get_arctic_data(start=start, end='', tempo='5S')
        
        while not data_loaded:
            try:
                weekday_bef = pd.to_datetime(start).weekday()
                if weekday_bef == 5: start = str(pd.to_datetime(start).date() - timedelta(days=1))
                add = ' 00' if weekday_bef in [0, 1, 2, 3, 4] else ' 18'; start += add
                self.print(start)
                data = self.get_arctic_data(start=start, end='', tempo='5S')
                data_loaded = True
                self.print('Data Loaded')
            except: 
                data_loaded = False
                start = start[:-3]
                start = str(pd.to_datetime(start).date() - timedelta(days=1))
                self.print('Trying again...')

        until = data.index[-1]
        timezone =  'UTC'#str(get_localzone()) if system()=='Windows' else 'UTC'
        data_last = self.download_complete(until, timezone)
        data_last.to_json('ib_data/%s_ib_data.json'%self.symbol); data_last.to_csv('ib_data/%s_ib_data.csv'%self.symbol)
        data = pd.concat([data, data_last])

        bars_df = util.df(self.bars)[['time', 'open_', 'high', 'low', 'close', 'volume']]
        bars_df.time = pd.to_datetime(bars_df.time).dt.tz_convert('US/Eastern').dt.tz_localize(None)
        bars_df.set_index('time', inplace=True)
        bars_df.index.names = ['time']
        bars_df.columns = ['open', 'high', 'low', 'close', 'volume']
        data = pd.concat([data, bars_df.loc[data.index[-1]:].iloc[1:]])
        #data = data.iloc[-20:]
        data.to_json('historical_data/%s_historical_data.json'%self.symbol)
        data.to_csv('historical_data/%s_historical_data.csv'%self.symbol)

        return data

    def last_bar_func(self, last_date):
        '''Last bar of streaming data

        Parameters:
            last_date (str): date of last bar
        
        Returns:
            DataFrame: last bar of streaming data
        '''
        date = pd.to_datetime(str(self.bars[-1].time)).tz_convert('US/Eastern').tz_localize(None)
        if date - last_date > timedelta(seconds=30):
            bars_df = util.df(self.bars)[['time', 'open_', 'high', 'low', 'close', 'volume']].set_index('time')
            bars_df.columns = ['open', 'high', 'low', 'close', 'volume']
            bars_df.index = bars_df.index.tz_convert('US/Eastern').tz_localize(None)
            bars_df[['open', 'high', 'low', 'close']] = bars_df[['open', 'high', 'low', 'close']].apply(self.x_round)
            last_bar = bars_df.loc[last_date:].iloc[1:]
        else:
            open_ = self.x_round(self.bars[-1].open_)
            high = self.x_round(self.bars[-1].high)
            low = self.x_round(self.bars[-1].low)
            close = self.x_round(self.bars[-1].close)
            volume = self.bars[-1].volume
            last_bar = pd.DataFrame({'date':date, 'open':open_, 'high':high, 'low':low, 'close':close, 'volume':volume}, index=['0']).set_index('date')
        return last_bar

    def save_data_csv(self, data_path):#'%s_data.csv'%(self.symbol)
        if not path.exists(data_path):
            results_file = open(data_path,'w')
        else: results_file = open(data_path,'a')
        bar = ['%s,'%(pd.to_datetime(self.bars[-1].time).tz_convert('US/Eastern').tz_localize(None)), '%.1f,'%(self.bars[-1].open_),
            '%.1f,'%(self.bars[-1].high), '%.1f,'%(self.bars[-1].low), '%.1f,'%(self.bars[-1].close), '%d'%(self.bars[-1].volume), '\n']
        results_file.writelines(bar)
        results_file.close()
    
    def save_last_bar(self):
        date = pd.to_datetime(str(self.bars[-1].time)).tz_convert('US/Eastern').tz_localize(None)
        open_ = self.x_round(self.bars[-1].open_)
        high = self.x_round(self.bars[-1].high)
        low = self.x_round(self.bars[-1].low)
        close = self.x_round(self.bars[-1].close)
        volume = self.bars[-1].volume
        last_bar = pd.DataFrame({'date':date, 'open':open_, 'high':high, 'low':low, 'close':close, 'volume':volume}, index=['0']).set_index('date')
        last_bar.to_csv('last_bar/%s_last_bar.csv'%self.symbol)
        last_bar.to_json('last_bar/%s_last_bar.json'%self.symbol)

    def reconnection(self):
        '''Disconnection and reconnection in platform and market closing
        '''
        if self.hour == '23:44:30' or self.hour == '16:59:30':
            self.interrumption = True
            self.ib.disconnect()
            self.connected = self.ib.isConnected()
            self.print('%s %s | Ib disconnection' % (self.date, self.hour))
            self.print('Connected: %s' % self.connected)
        if self.hour == '23:46:00' or self.hour == '18:00:05':
            self.interrumption = False
            self.print('%s %s | Reconnecting...' % (self.date, self.hour))
            while not self.connected:
                try:
                    self.ib.connect('127.0.0.1', self.port, self.client)
                    self.connected = self.ib.isConnected()
                    if self.connected:
                        self.print('%s %s | Connection reestablished!' % (self.date, self.hour))
                        self.print('Requesting Market Data...')
                        self.bars = self.ib.reqRealTimeBars(self.contract, 5, 'TRADES', False)
                        self.print('Last Close of %s: %.2f' % (self.symbol, self.bars[-1].close))
                        self.print('%s Data has been Updated!' % self.symbol)
                except:
                    self.print('%s %s | Connection Failed! Trying to reconnect in 10 seconds...' % (self.date, self.hour))
                    self.ib.sleep(10)
            self.print('%s %s | %s Data has been Updated!' % (self.date, self.hour, self.symbol))
    
    def send_telegram_message(self, message, type='data'): #
        '''Send telegram message to an specific group

        Parameters:
            message (string): Message to be sent
            type (string): if 'trades' sends message to trades telegram group. if 'info' sends message to information telegram group
        '''
        telegram_credentials = pd.read_csv('telegram_credentials.csv')

        bot_token = telegram_credentials['bot_token'].iloc[0]                  # bot token
        chatID_data = telegram_credentials['bot_chatID_data'].iloc[0]          # chat ID data
        chatID_alarm = telegram_credentials['bot_chatID_alarm'].iloc[0]        # chat ID alarm
        bot_chatID = chatID_data if type=='data' else chatID_alarm
        url = 'https://api.telegram.org/bot%s/sendMessage?chat_id=%s&text=%s'%(bot_token, bot_chatID, message)
    
        requests.get(url)
    
    def resampler(self, data, tempo, type='ticks'):
        '''Resample data according to type of bars

        Parameters:
            data (DataFrame): data to resample
            tempo (string): temporality of resulting resampled data
            type (string): type of entry data
        
        Returns:
            DataFrame: resampled data according to type and tempo
        '''
        col_names = ['Last', 'Last', 'Last', 'Last', 'Volume'] if type == 'ticks' else ['open', 'high', 'low', 'close', 'volume']
        Open = data[col_names[0]].resample(tempo).first()
        High = data[col_names[1]].resample(tempo).max()
        Low = data[col_names[2]].resample(tempo).min()
        Close = data[col_names[3]].resample(tempo).last()
        Volume = data[col_names[4]].resample(tempo).sum()
        resampled_data = pd.concat([Open, High, Low, Close, Volume],axis=1).dropna()
        resampled_data.columns = ['open', 'high', 'low', 'close', 'volume']
        return resampled_data