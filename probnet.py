import numpy as np
from matplotlib import pyplot as plt
import itertools
import statsmodels.api as sm
import pandas as pd
from fbprophet import Prophet
flow_file = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/flow_train.csv'
df = pd.read_csv(flow_file)
df['date_dt'] = pd.to_datetime(df['date_dt'].astype(str), format='%Y%m%d')
flow_group = df.groupby(by='district_code')


target = np.zeros(shape=(98, 274, 3),dtype=np.float32)

for ix, x in enumerate(flow_group.count().index):

    d = df[df['district_code']==x]
    ds = d[['date_dt', 'dwell' ]]
    #ds = ds.set_index('date_dt')
    break

furniture = ds.rename(columns={'date_dt': 'ds', 'dwell': 'y'})
print(furniture)
furniture_model = Prophet(interval_width=0.95)
furniture_model.fit(furniture)
furniture_forecast = furniture_model.make_future_dataframe(periods=1, freq='MS')
furniture_forecast = furniture_model.predict(furniture_forecast)
furniture_model.plot(furniture_forecast, xlabel = 'Date', ylabel = 'dwell')
plt.title('Furniture dwell')
plt.show()

