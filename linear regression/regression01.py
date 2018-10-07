import pandas as pd
import quandl, math
import numpy as np
import matplotlib.pyplot as plot
from matplotlib import style
import datetime
import time
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['high_low_pct'] = (df['Adj. High'] - df['Adj. Low'])/df['Adj. Close'] * 100.0
df['pct_change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'high_low_pct', 'pct_change', 'Adj. Volume']]
df.fillna(-99999, inplace=True)

forecast_column = 'Adj. Close'
forecast_interval = int(math.ceil(0.005*len(df)))
df['label'] = df[forecast_column].shift(-forecast_interval)

#creating test and training set
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_predict = X[-forecast_interval:]
X = X[:-forecast_interval]

df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
classifier = LinearRegression()
classifier.fit(X_train, y_train)
accuracy = classifier.score(X_test, y_test)
print(accuracy)

#predict data
predicted_result = classifier.predict(X_predict)
print(predicted_result)

#plot result in graph
last_date = df.iloc[-1].name
# last_unix = last_date.timestamp()
last_unix = time.mktime(last_date.timetuple())
day = 24*60*60
next_unix = last_unix + day

df['prediction'] = np.nan
for i in predicted_result:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['Adj. Close'].plot()
df['prediction'].plot()
plot.legend(loc = 4)
plot.xlabel('Date')
plot.ylabel('Price')
plot.show()

# print(df.head)

