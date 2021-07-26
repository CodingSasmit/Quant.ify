import oandapyV20
import pandas as pd
import numpy as np
from oandapyV20.contrib.factories import InstrumentsCandlesFactory
import pytz, csv
from datetime import datetime

fields = ['time', 'open', 'high', 'low', 'close']
def heiken_ashi(df):
    df['h-open']=0
    df.loc[1:, 'h-open'] = np.mean([df.loc[:len(df)-2, 'open'], df.loc[:len(df)-2, 'close']], axis=0)
    df['h-high'] = np.max([df['open'], df['high'], df['close']], axis=0)
    df['h-low'] = np.min([df['open'], df['low'], df['close']], axis=0)
    df['h-close'] = np.mean([df['open'], df['high'], df['low'], df['close']], axis=0)
    df['h-color'] = (df['h-close']>df['h-open']).astype(int)
def candle_color(df):
    df['candle_color'] = (df['close']>df['open']).astype(int)
def trends(df):
    df['short_term_trend'] = (df['close']>df['ema-50']).astype(int)
    df['mid_term_trend'] = (df['close']>df['ema-100']).astype(int)
    df['long_term_trend'] = (df['close']>df['ema-200']).astype(int)
def time_of_day_week(df):
    df['time_of_day'] = df.loc[:, 'time'].str[11:13]
    df['time_of_day'] = df['time_of_day'].astype(int)
    Y=df.loc[:, 'time'].str[:4]
    M=df.loc[:, 'time'].str[5:7]
    D=df.loc[:, 'time'].str[8:10]
    df['day_of_week'] = [datetime(year=int(y), month=int(m), day=int(d)).weekday() for y, m, d in zip(Y, M, D)]
def get_data(instrument, params, filename, client):
    reset_file(filename)
    for i in InstrumentsCandlesFactory(instrument=instrument, params=params):
        rv = client.request(i)
        data_to_csv(filename, i.response)
    return
def data_to_csv(file_name, r):
    with open(file_name, mode='a+') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        for candle in r.get('candles'):
            values = {
                'time': candle['time'][:19]+'Z',
                'open': candle['mid']['o'],
                'high': candle['mid']['h'],
                'low': candle['mid']['l'],
                'close': candle['mid']['c'],
            }
            writer.writerow(values)
    return
def reset_file(file_name):
    file = open(file_name, 'w')
    file.truncate(0)
    file.close()
    with open(file_name, mode='a+') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
    return
def cci(df, period=20):
    df['TP'] = (df['high']+df['low']+df['close'])/3
    df['TP_SMA']=0.0
    df['CCI']=0.0
    for i in range(period, len(df)):
        df.at[i, 'TP_SMA'] = np.mean(df.loc[i-period:i, 'TP'])
        mean_deviation = np.mean(np.abs(df.at[i, 'TP_SMA'] - df.loc[i-period:i, 'TP']))
        df.at[i, 'CCI'] = (df.at[i, 'TP'] - df.at[i, 'TP_SMA'])/(1.5 * mean_deviation)

def stochastic(df, period=14): # same as %K
    df['%K'] = 0.0
    for i in range(period, len(df)):
        high = np.max(df.loc[i-period:i, 'high'])
        low = np.min(df.loc[i-period:i, 'low'])
        close = df.at[i, 'close']
        df.at[i, '%K'] = (close-low)/(high-low)
def rsi(df, period=14):
    df['RSI'], df['mean_gain'], df['mean_loss'] = 0.0, 0.0, 0.0
    gain, loss = 0, 0
    for j in range(0, period):
        change = df.at[j+1, 'close']/df.at[j, 'close']
        if change>=1:
            gain += change-1
        else:
            loss += 1-change
    df.at[period, 'mean_gain'] = gain/period
    df.at[period, 'mean_loss'] = loss/period
    RS = gain/loss
    df.at[period, 'RSI'] = 1 - 1/(1+RS)
    for i in range(period+1, len(df)):
        gain, loss = 0, 0
        change = df.at[i, 'close']/df.at[i-1, 'close']
        if change>=1:
            gain += change-1
        else:
            loss += 1-change
        df.at[i, 'mean_gain'] = (df.at[i-1, 'mean_gain']*13 + gain)/period
        df.at[i, 'mean_loss'] = (df.at[i-1, 'mean_loss']*13 + loss)/period
        RS = df.at[i, 'mean_gain']/df.at[i, 'mean_loss']
        df.at[i, 'RSI'] = 1 - 1/(1+RS)
def ema(df, period):
    s = 'ema-{}'.format(period)
    df[s]=0.0
    df.at[period, s] = np.mean(df.loc[:period, 'close'])
    multiplier = 2/(1+period)
    for i in range(period+1, len(df)):
        df.at[i, s] = df.at[i, 'close']*multiplier + df.at[i-1, s]*(1-multiplier)

def atr(df, period=14):
    df['ATR']=0.0
    df['TR']=0.0
    for i in range(1, len(df)):
        h, l, c = df.at[i, 'high'], df.at[i, 'low'], df.at[i-1, 'close']
        df.at[i, 'TR'] = np.max([h-l, abs(h-c), abs(l-c)])
    for i in range(period, len(df)):
        df.at[i, 'ATR'] = np.mean(df.loc[i-period+1:i+1, 'TR'])
def add_indicators(df):
    candle_color(df)
    heiken_ashi(df)
    cci(df)
    stochastic(df)
    rsi(df)
    for i in [12, 26, 50, 100, 200]:
        ema(df, i)
    df['macd'] = df['ema-12']-df['ema-26']
    s = 'signal'
    df[s]=0.0
    multiplier = 2/(1+9)
    df.at[40, s] = np.mean(df.loc[31:40, 'ema-12']-df.loc[31:40, 'ema-26'])
    for i in range(41, len(df)):
        df.at[i, s] = (df.at[i, 'ema-12']-df.at[i, 'ema-26'])*multiplier + df.at[i-1, s]*(1-multiplier)
    atr(df)
    time_of_day_week(df)
    trends(df)
