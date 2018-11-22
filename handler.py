import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
flow_file = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/flow_train.csv'
flow_data = pd.read_csv(flow_file)
print(flow_data.describe())
flow_group = flow_data.groupby(by='district_code')
print(flow_group.count())
d = flow_data[flow_data['district_code']=='ac12452bc2dff4b1d376ed517b9f74f4']
dates = d['date_dt'].values

xs = [datetime.strptime(str(d), '%Y%m%d').date() for d in dates]

dwell = d['dwell'].values
flow_in = d['flow_in'].values
flow_out = d['flow_out'].values
np.save('data',np.asarray([dwell,flow_in,flow_out]))


plt.plot(xs, dwell)
plt.plot(xs, flow_in)
plt.plot(xs, flow_out)
plt.show()
