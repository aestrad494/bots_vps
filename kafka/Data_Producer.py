from kafka import KafkaProducer
import json
from time import sleep
from datetime import datetime
import pandas as pd

class producer():
    def __init__(self, symbol):
        self.symbol = symbol
        self.producer = KafkaProducer(bootstrap_servers=['165.227.192.87:9092'], value_serializer=self.json_serializer)
        self.current_date()

    def current_date(self):
        '''Get current date, weekday and hour
        '''
        self.date = datetime.now().strftime('%Y-%m-%d')
        self.weekday = datetime.now().weekday()
        self.hour = datetime.now().strftime('%H:%M:%S')
    
    def json_serializer(self, data):
        return json.dumps(data).encode("utf-8")

    def run_producer(self):
        while self.weekday in [0, 1, 2, 3, 4, 6] and not (self.weekday == 4 and pd.to_datetime(self.hour).time() > pd.to_datetime('17:10:00').time()):
            self.current_date()

            with open('last_bar/%s_last_bar.json'%self.symbol) as json_file:
                last_bar = json.load(json_file)
                print('new bar')
            
            self.producer.send('%s_json'%self.symbol, last_bar)
            sleep(5)

data_producer = producer('M2K')
data_producer.run_producer()