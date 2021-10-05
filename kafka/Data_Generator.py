#
# Python Script with live trading 
# Hermes Strategy Class
# 
# (c) Andres Estrada Cadavid
# QSociety

from Generator import Generator
import pandas as pd
from datetime import datetime

class LiveHermes(Generator):
    def run_strategy(self):
        self.print('%s %s | %s Bot Turned On' % (self.date, self.hour, self.bot_name))

        # Check if operable schedule
        self.operable_schedule()

        if self.operable:
            # Getting Historical Data to get renko bars
            self.print('Downloading Historical %s Data...'%self.symbol)
            self.data = self.download_historical(days_before=14)
            self.print('Historical Data retrieved! from %s to %s'%(self.data.index[0], self.data.index[-1]))
            self.print(self.data.tail())

            while self.weekday in [0, 1, 2, 3, 4, 6] and not (self.weekday == 4 and pd.to_datetime(self.hour).time() > pd.to_datetime('17:10:00').time()):
                # Concatening last bar
                if datetime.now().second % 5 == 0:
                    last_bar = self.last_bar_func(last_date=self.data.index[-1])
                    if self.data.index[-1] != last_bar.index[-1]:
                        self.data = pd.concat([self.data, last_bar])
                        self.save_data_csv('historical_data/%s_historical_data.csv'%self.symbol)
                        self.save_data_csv('ib_data/%s_ib_data.csv'%self.symbol)
                        #self.data.to_json('historical_data/%s_historical_data.json'%self.symbol)
                        self.save_last_bar()
                        self.print(self.data.iloc[[-1]])
                
                self.current_date()
                self.ib.sleep(1)
                if len(self.bars) > 0 and self.bars[-1].close > 0: 
                    # Send confirmation message each hour to telegram
                    self.continuous_check_message('%s %s | %s %s is running OK. Last price: %.2f' % 
                                                (self.date, self.hour, self.bot_name, self.symbol, self.x_round(self.bars[-1].close)))

                # Check if it's time to reconnect
                self.connected = self.ib.isConnected()
                self.reconnection()

                if not self.connected:
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
                                self.data = self.download_historical(days_before=14)
                                self.bars = self.ib.reqRealTimeBars(self.contract, 5, 'TRADES', False)
                                self.print('Last Close of %s: %.2f' % (self.symbol, self.bars[-1].close))
                                self.print('%s Data has been Updated!' % self.symbol)
                        except:
                            self.print('Connection Failed! Trying to reconnect in 10 seconds...')

            self.ib.disconnect()
            self.print('%s %s | Session Ended. Good Bye!' % (self.date, self.hour))

if __name__ == '__main__':
    symbol = input('\tSymbol: ')#'M2K'
    port = 7497
    client = 115

    live_hermes = LiveHermes(symbol=symbol, bot_name='Data Generator', port=port, client=client)
    live_hermes.run_strategy()