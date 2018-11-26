import pandas as pd
import numpy as np
import csv
times = ['20180302','20180303','20180304','20180305','20180306',
         '20180307','20180308','20180309','20180310','20180311',
         '20180312','20180313','20180314','20180315','20180316']
flow_file = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/flow_train.csv'
flow_data = pd.read_csv(flow_file)
flow_group = flow_data.groupby(by='district_code')

target = np.zeros(shape=(98, 274, 3),dtype=np.float32)

f = open('prediction.csv','w')
writer = csv.writer(f)

for ix, x in enumerate(flow_group.count().index):
    d = flow_data[flow_data['district_code'] == x]
    city_code = d['city_code'].values[0]
    district_code = x
    data = np.load('log/'+str(ix)+'.npy')
    for t, flow in enumerate(times):
        l = [flow, city_code, district_code,data[t][0],data[t][1],data[t][2]]
        writer.writerow(l)
        f.flush()



