import oandapyV20
import pandas as pd
import numpy as np
from oandapyV20.contrib.factories import InstrumentsCandlesFactory
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.orders as orders
from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, \
    StopLossDetails, PositionCloseRequest, TradeCloseRequest
import oandapyV20.types as types
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.trades as trades
import pytz, csv
from datetime import timedelta, datetime
import time, gc, requests, os, json
import keras, tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from heiken_functions import *
from googleapiclient.discovery import build
from google.oauth2 import service_account

FORMAT = "%Y-%m-%dT%H:%M:%SZ"
# ACCOUNT = '101-001-17686216-003'
UTC = pytz.utc
PRICE_FEATURES = ['TP', 'TP_SMA', 'ema-12', 'ema-26', 'ema-50', 'ema-100',
                  'ema-200']
ORIG_FEATURES = ['open', 'high', 'low', 'close', 'h-open', 'h-high', 'h-low', 'h-close']

service_account_file = 'google_key.json'
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
credentials = service_account.Credentials.from_service_account_file(service_account_file, scopes=SCOPES)

SPREADSHEET_ID = 'SPREADSHEET ID'
service = build('sheets', 'v4', credentials=credentials)
sheet = service.spreadsheets()

class SingleCurrencyTrader:
    client = oandapyV20.API(access_token='USE_OWN_ACCESS_TOKEN')
    def __init__(self, name, instrument, account, cutoff, max_spread, model_path):
        self.name = name
        self.instrument = instrument
        self.account = account
        self.model_path = model_path
        r = positions.PositionDetails(accountID=self.account, instrument=instrument)
        try:
            r = self.client.request(r)
            self.pl = float(r['position']['pl'])
        except:
            self.pl = 0
        self.h4 = pd.DataFrame()
        self.m15 = pd.DataFrame()
        self.model = Model()
        self.cutoff = cutoff
        self.max_spread = max_spread
        try:
            file = open('json/'+self.name+'.json', 'r')
            x=json.load(file)
            file.close()
            self.position = x['position']
            self.elapsed = x['elapsed']
            self.actual_pl = x['actual_pl']
            self.total_trades, self.successful_trades = x['total_trades'], x['successful_trades']
        except:
            self.position = False
            self.elapsed = 0
            self.actual_pl = 0
            self.total_trades, self.successful_trades = 0, 0


    def model_init(self):
        input1 = Input(shape=(180, 5))
        x = LSTM(180)(input1)
        x = Dropout(0.2)(x)
        branch1 = Model(inputs=input1, outputs=x)

        input2 = Input(shape=(240, 10))
        y = LSTM(240)(input2)
        y = Dropout(0.2)(y)
        branch2 = Model(inputs=input2, outputs=y)

        input3 = Input(shape=18)
        a = Dense(128, activation=keras.layers.LeakyReLU(alpha=0.1))(input3)
        a = Dropout(0.2)(a)
        features_branch = Model(inputs=input3, outputs=a)

        combined = concatenate([branch1.output, branch2.output, features_branch.output])

        c = Dense(256, activation=keras.layers.LeakyReLU(alpha=0.1))(combined)
        c = Dropout(0.2)(c)
        c = Dense(3, activation="softmax", name='30_pip')(c)

        model = Model(inputs=[branch1.input, branch2.input, features_branch.input], outputs=c)
        return model

    def init(self):
        self.model = self.model_init()
        self.model.load_weights(self.model_path).expect_partial()

        now = datetime.now(UTC) - timedelta(seconds=4)
        end = now.strftime(FORMAT)
        start = (now - timedelta(days=45)).strftime(FORMAT)

        params = {"from": start, "to": end, "granularity": "H4", "count": 5000}

        get_data(self.instrument, params, 'h4_candles.csv', self.client)

        params['from'] = (now - timedelta(days=19)).strftime(FORMAT)
        params['granularity'] = 'H1'
        get_data(self.instrument, params, 'm15_candles.csv', self.client)

        # reading values into dataframes and adding indicators
        self.h4 = pd.read_csv("h4_candles.csv")
        self.m15 = pd.read_csv('m15_candles.csv')

        self.h4[['open', 'high', 'low', 'close']] = self.h4[['open', 'high', 'low', 'close']].astype(np.float64)
        self.m15[['open', 'high', 'low', 'close']] = self.m15[['open', 'high', 'low', 'close']].astype(np.float64)

        add_indicators(self.m15)
        candle_color(self.h4)

        self.h4 = self.h4[-180:]
        self.m15 = self.m15[-240:]

        self.h4.reset_index(inplace=True, drop=True)
        self.m15.reset_index(inplace=True, drop=True)
    def update(self):
        now = datetime.now(UTC) - timedelta(seconds=4)

        params = {"from": now.strftime(FORMAT), "granularity": "H4", "count": 1}
        new = self.data_batch(params)
        self.df_update(self.h4, new)
        j=len(self.h4)
        self.h4.at[j-1, 'candle_color'] = (self.h4.at[j-1, 'close']>self.h4.at[j-1, 'open']).astype(int)

        params['granularity'] = 'H1'
        new = self.data_batch(params)
        self.df_update(self.m15, new)
        self.lastrow_indicators(self.m15)
    def json_update(self):
        x = {
            'successful_trades': self.successful_trades,
            'total_trades': self.total_trades,
            'actual_pl': self.actual_pl,
            'position': self.position,
            'elapsed': self.elapsed
        }
        with open('json/'+self.name+'.json', 'w') as file:
            file.write(json.dumps(x))
    def df_update(self, df, new):
        if len(new)>0 and new[0]['time']==df.loc[len(df)-1, 'time']:
            df.at[len(df)-1, ['time', 'open', 'high', 'low', 'close']] = new[0].values()
        elif len(new)>0 and new[0]['time']!=df.loc[len(df)-1, 'time']:
            df.at[len(df), ['time', 'open', 'high', 'low', 'close']] = new[0].values()
            df.drop(df.index[0], inplace=True)
        df.reset_index(inplace=True, drop=True)

    def lastrow_indicators(self, df):
        i = len(df)-1
        # heiken ashi
        df.at[i, 'h-open'] = np.mean([df.at[i-1, 'open'], df.at[i-1, 'close']])
        df.at[i, 'h-high'] = np.max([df.at[i, 'open'], df.at[i, 'high'], df.at[i, 'close']], axis=0)
        df.at[i, 'h-low'] = np.min([df.at[i, 'open'], df.at[i, 'low'], df.at[i, 'close']], axis=0)
        df.at[i, 'h-close'] = np.mean([df.at[i, 'open'], df.at[i, 'high'], df.at[i, 'low'], df.at[i, 'close']], axis=0)
        df.at[i, 'h-color'] = (df.at[i, 'h-close']>df.at[i, 'h-open']).astype(int)
        # candle color
        df.at[i, 'candle_color'] = (df.at[i, 'close']>df.at[i, 'open']).astype(int)
        #cci
        df.at[i, 'TP'] = (df.at[i, 'high']+df.at[i, 'low']+df.at[i, 'close'])/3
        df.at[i, 'TP_SMA'] = np.mean(df.loc[i-20:i, 'TP'])
        mean_deviation = np.mean(np.abs(df.at[i, 'TP_SMA'] - df.loc[i-20:i, 'TP']))
        df.at[i, 'CCI'] = (df.at[i, 'TP'] - df.at[i, 'TP_SMA'])/(1.5 * mean_deviation)

        # stochastic
        high = np.max(df.loc[i-14:i, 'high'])
        low = np.min(df.loc[i-14:i, 'low'])
        close = df.at[i, 'close']
        df.at[i, '%K'] = (close-low)/(high-low)

        # rsi
        gain, loss = 0, 0
        change = df.at[i, 'close']/df.at[i-1, 'close']
        if change>=1:
            gain += change-1
        else:
            loss += 1-change
        df.at[i, 'mean_gain'] = (df.at[i-1, 'mean_gain']*13 + gain)/14
        df.at[i, 'mean_loss'] = (df.at[i-1, 'mean_loss']*13 + loss)/14
        RS = df.at[i, 'mean_gain']/df.at[i, 'mean_loss']
        df.at[i, 'RSI'] = 1 - 1/(1+RS)

        # ema 5, 10, 20
        for j in [12, 26, 50, 100, 200]:
            n = 'ema-'+str(j)
            multiplier = 2/(1+j)
            df.at[i, n] = df.at[i, 'close']*multiplier + df.at[i-1, n]*(1-multiplier)
        df.at[i, 'macd'] = df.at[i, 'ema-12']-df.at[i, 'ema-26']

        multiplier = 2/(1+9)
        df.at[i, 'signal'] = df.at[i, 'macd']*multiplier + df.at[i-1, 'signal']*(1-multiplier)
        # atr
        h, l, c = df.at[i, 'high'], df.at[i, 'low'], df.at[i-1, 'close']
        df.at[i, 'TR'] = np.max([h-l, abs(h-c), abs(l-c)])
        df.at[i, 'ATR'] = np.mean(df.loc[i-6+1:i+1, 'TR'])

        # trends
        df.at[i, 'short_term_trend'] = (df.at[i, 'close']>df.at[i, 'ema-50']).astype(int)
        df.at[i, 'mid_term_trend'] = (df.at[i, 'close']>df.at[i, 'ema-100']).astype(int)
        df.at[i, 'long_term_trend'] = (df.at[i, 'close']>df.at[i, 'ema-200']).astype(int)

        # time and day of week
        df.at[i, 'time_of_day'] = int(df.at[i, 'time'][11:13])
        df.at[i, 'day_of_week'] = datetime(year=int(df.at[i, 'time'][:4]), month=int(df.at[i, 'time'][5:7]), day=int(df.at[i, 'time'][8:10])).weekday()

    def data_batch(self, params):
        r = instruments.InstrumentsCandles(self.instrument, params)
        r = self.client.request(r)
        new = []
        for i in range(len(r['candles'])):
            c = r['candles'][i]['mid']
            new.append({
                'time': r['candles'][i]['time'][:19],
                'open': float(c['o']),
                'high': float(c['h']),
                'low': float(c['l']),
                'close': float(c['c'])})
        return new

    def predict(self):
        h = self.h4[['open', 'high', 'low', 'close', 'candle_color']].copy()
        m = self.m15[['open', 'high', 'low', 'close', 'candle_color', 'h-open', 'h-high', 'h-low', 'h-close', 'h-color']].copy()
        f = self.m15[['TP', 'TP_SMA', 'CCI', '%K', 'RSI', 'ema-12', 'ema-26', 'ema-50', 'ema-100', 'ema-200', 'macd',
                      'signal', 'ATR', 'time_of_day', 'day_of_week', 'short_term_trend', 'mid_term_trend', 'long_term_trend']].copy()

        f[['macd', 'signal', 'ATR']] *= 1000

        # h[['open', 'high', 'low', 'close']] = (h[['open', 'high', 'low', 'close']]-1)*10
        # m[ORIG_FEATURES] = (m[ORIG_FEATURES]-1)*10
        # f[PRICE_FEATURES] = (f[PRICE_FEATURES]-1)*10

        f = f[-1:]

        h = np.array([h]).astype(np.float64)
        m = np.array([m]).astype(np.float64)
        f = np.array(f).astype(np.float64)

        # trading based on predicted values
        return self.model.predict([h, m, f])[0]
    def place_orders(self, pred):
        r = pricing.PricingInfo(accountID=self.account, params={'instruments': self.instrument})
        r=self.client.request(r)
        spread = (float(r['prices'][0]['closeoutAsk'])-float(r['prices'][0]['closeoutBid']))*1e4
        if spread>self.max_spread:
            return

        r = pricing.PricingInfo(accountID=self.account, params={'instruments': self.instrument})
        r=self.client.request(r)
        ask = float(r['prices'][0]['asks'][0]['price'])
        bid = float(r['prices'][0]['bids'][0]['price'])

        r2 = accounts.AccountSummary(accountID=self.account)
        r2 = self.client.request(r2)
        balance = float(r2['account']['balance'])*0.25

        if pred[2]>self.cutoff:
            print('went long')
            mktOrder = MarketOrderRequest(instrument=self.instrument,
                                          units=int(balance*50/ask),
                                          ).data
            r = orders.OrderCreate(accountID=self.account, data=mktOrder)
            r = self.client.request(r)
            self.position=True
        elif pred[0]>self.cutoff:
            print('went short')
            mktOrder = MarketOrderRequest(instrument=self.instrument,
                                          units=-int(balance*50/bid),
                                          ).data
            r = orders.OrderCreate(accountID=self.account, data=mktOrder)
            r = self.client.request(r)
            self.position=True
    def monitor_trade(self):
        if not self.position:
            return
        if self.elapsed==5:
            self.elapsed=0
            r = trades.OpenTrades(self.account)
            r = self.client.request(r)
            id = r['trades'][0]['id']
            p = TradeCloseRequest(units='ALL').data
            r = trades.TradeClose(accountID=self.account, tradeID=id, data=p)
            r = self.client.request(r)
            self.position = False

            r=positions.PositionDetails(accountID=self.account, instrument=self.instrument)
            r = self.client.request(r)
            new_pl = float(r['position']['pl'])
            if new_pl!=self.pl:
                self.actual_pl += new_pl-self.pl
                self.total_trades+=1
            if new_pl-self.pl>0:
                self.successful_trades += 1
            self.pl = new_pl
        else:
            self.elapsed+=1

    def output(self, pred):
        wr = float(i.successful_trades)/i.total_trades if i.total_trades>0 else 0
        v = [self.name, float(self.cutoff), float(pred[0]),
             float(pred[2]), float(self.successful_trades), float(self.total_trades),
              wr, float(self.actual_pl)]
        return v

pairs = [
    SingleCurrencyTrader('.320', 'EUR_USD', '101-001-17686216-002', 0.36, 1.8, 'models/heiken_10_year/m.320'),
    SingleCurrencyTrader('.290', 'EUR_USD', '101-001-17686216-003', 0.36, 1.8, 'models/heiken_10_year/m.290'),
    SingleCurrencyTrader('.210', 'EUR_USD', '101-001-17686216-004', 0.345, 1.8, 'models/heiken_10_year/m.210'),
    SingleCurrencyTrader('.190', 'EUR_USD', '101-001-17686216-005', 0.35, 1.8, 'models/heiken_10_year/m.190'),
    SingleCurrencyTrader('.90', 'EUR_USD', '101-001-17686216-006', 0.33, 1.8, 'models/heiken_10_year/m.090'),
    SingleCurrencyTrader('.100', 'EUR_USD', '101-001-17686216-007', 0.35, 1.8, 'models/heiken_10_year/m.100'),
    SingleCurrencyTrader('.250', 'EUR_USD', '101-001-17686216-008', 0.36, 1.8, 'models/heiken_10_year/m.250'),
    SingleCurrencyTrader('.170', 'EUR_USD', '101-001-17686216-009', 0.36, 1.8, 'models/heiken_10_year/m.170'),
    SingleCurrencyTrader('.260', 'EUR_USD', '101-001-17686216-010', 0.35, 1.8, 'models/heiken_10_year/m.260'),
    SingleCurrencyTrader('.400', 'EUR_USD', '101-001-17686216-011', 0.34, 1.8, 'models/heiken_10_year/m.400'),
    SingleCurrencyTrader('.360', 'EUR_USD', '101-001-17686216-012', 0.355, 1.8, 'models/heiken_10_year/m.360'),
    SingleCurrencyTrader('.300', 'EUR_USD', '101-001-17686216-013', 0.34, 1.8, 'models/heiken_10_year/m.300'),
    SingleCurrencyTrader('.230', 'EUR_USD', '101-001-17686216-014', 0.345, 1.8, 'models/heiken_10_year/m.230'),
    SingleCurrencyTrader('sh.100', 'EUR_USD', '101-001-17686216-015', 0.35, 1.8, 'models/10_year_shuffled/shm.100'),
    SingleCurrencyTrader('sh.200', 'EUR_USD', '101-001-17686216-016', 0.36, 1.8, 'models/10_year_shuffled/shm.200'),
    SingleCurrencyTrader('sh.300', 'EUR_USD', '101-001-17686216-017', 0.36, 1.8, 'models/10_year_shuffled/shm.300'),
    SingleCurrencyTrader('sh.400', 'EUR_USD', '101-001-17686216-018', 0.36, 1.8, 'models/10_year_shuffled/shm.400'),
    SingleCurrencyTrader('sh.500', 'EUR_USD', '101-001-17686216-019', 0.36, 1.8, 'models/10_year_shuffled/shm.500')
]
for i in pairs:
    i.init()
hour = False

while True:
    now = datetime.now(UTC) - timedelta(seconds=5)
    if now.minute==0:
        hour=False
    for i in pairs:
        i.update()
    pred = [i.predict() for i in pairs]

    v = [['Current Time: {}\n'.format(now.strftime('%H:%M:%S'))]]
    sheet.values().update(spreadsheetId=SPREADSHEET_ID, range='Sheet1!J1',
                          valueInputOption='USER_ENTERED', body={'values': v}).execute()
    val = []
    for i, p in zip(pairs, pred):
        val.append(i.output(p))
    sheet.values().update(spreadsheetId=SPREADSHEET_ID, range='Sheet1!A2',
                        valueInputOption='USER_ENTERED', body={'values': val}).execute()

    not_late_friday_pos = now.weekday()<4 or (now.weekday()==4 and now.hour<17) or (now.weekday()==6 and now.hour>=21)
    if now.minute==59 and now.second>30 and not hour:
        hour=True
        if not_late_friday_pos:
            for i, p in zip(pairs, pred):
                if not i.position:
                    i.place_orders(p)
        for i in pairs:
            i.monitor_trade()
            i.json_update()
    gc.collect()
    os.system('clear')
    print('running...')
    time.sleep(10)
