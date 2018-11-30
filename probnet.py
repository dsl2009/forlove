import numpy as np
from matplotlib import pyplot as plt
import itertools
import csv
import statsmodels.api as sm
import pandas as pd
from fbprophet import Prophet
flow_file = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/flow_train.csv'
df = pd.read_csv(flow_file)
df['date_dt'] = pd.to_datetime(df['date_dt'].astype(str), format='%Y%m%d')
flow_group = df.groupby(by='district_code')
print(pd.date_range(start='2017-06-01', end='2018-03-01'))
times = ['20180302','20180303','20180304','20180305','20180306',
         '20180307','20180308','20180309','20180310','20180311',
         '20180312','20180313','20180314','20180315','20180316']
target = np.zeros(shape=(98, 274, 3),dtype=np.float32)
f = open('prediction.csv','w')
writer = csv.writer(f)

for ix, x in enumerate(flow_group.count().index):

    d = df[df['district_code']==x]
    city_code = d['city_code'].values[0]
    district_code = x
    data = np.zeros((15,3),dtype=np.float32)
    for k, n in enumerate(['dwell', 'flow_in', 'flow_out']):
        ds = d[['date_dt', n ]]
        furniture = ds.rename(columns={'date_dt': 'ds', n: 'y'})
        furniture_model = Prophet()
        furniture_model.fit(furniture)
        furniture_forecast = furniture_model.make_future_dataframe(periods=15, freq='D')
        furniture_forecast = furniture_model.predict(furniture_forecast)
        pd = furniture_forecast.tail(15)[['ds', 'yhat']]
        data[:,k] = pd['yhat'].values
        furniture_model.plot(furniture_forecast, xlabel = 'Date', ylabel = 'dwell')
        furniture_model.plot_components(furniture_forecast)
        plt.title('Furniture dwell')
        plt.show()
    data[np.where(data<0)] = 10.0

    for t, flow in enumerate(times):
        l = [flow, city_code, district_code,data[t][0],data[t][1],data[t][2]]
        writer.writerow(l)
        f.flush()

