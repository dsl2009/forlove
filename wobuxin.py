import numpy as np
from matplotlib import pyplot as plt
from fbprophet import Prophet

import pandas as pd
d = pd.date_range(start='2017-06-01', end='2018-03-01').values

def shows():

    img = np.load('trans.npy')
    for i in range(98):
        for j in range(98):
            if i!=j:
                df = pd.DataFrame({'ds':d, 'y':img[:,i,j]})
                furniture_model = Prophet()
                furniture_model.fit(df)

                furniture_forecast = furniture_model.make_future_dataframe(periods=15, freq='D')
                furniture_forecast = furniture_model.predict(furniture_forecast)
                pds = furniture_forecast.tail(15)[['ds', 'yhat']]
                #furniture_model.plot(furniture_forecast, xlabel='Date', ylabel='dwell')

                furniture_model.plot_components(furniture_forecast)
                plt.title('Furniture dwell'+str(i)+str(j))
                plt.show()


shows()
