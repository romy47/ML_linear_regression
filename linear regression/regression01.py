import pandas as pd
import quandl
import math
df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['high_low_pct'] = (df['Adj. High'] - df['Adj. Low'])/df['Adj. Close'] * 100.0
df['pct_change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'high_low_pct', 'pct_change', 'Adj. Volume']]
df.fillna(-99999, inplace=True)

forecast_column = 'Adj. Close'
forecast_interval = int(math.ceil(0.005*len(df)))
df['label'] = df[forecast_column].shift(-forecast_interval)
df.dropna(inplace=True)
print(df.head)

