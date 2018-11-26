import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler

max_num = 6000
img = np.zeros(shape=[98, 274, 3],dtype=np.float32)
mean =[257.717463,    381.688098  , 381.688098]
std = [389.969604 ,696.205022 , 692.336528]
flow_file = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/flow_train.csv'
flow_data = pd.read_csv(flow_file)
flow_group = flow_data.groupby(by='district_code')

target = np.zeros(shape=(98, 274, 3),dtype=np.float32)



for ix, x in enumerate(flow_group.count().index):

    d = flow_data[flow_data['district_code']==x]
    dwell = d['dwell'].values
    flow_in = d['flow_in'].values
    flow_out = d['flow_out'].values

    dataes = np.vstack((dwell,flow_in,flow_out))
    dataes = np.transpose(dataes,axes=(1,0))
    target[ix] = dataes

    img[ix:(ix+1),:,:] = dataes
img = np.transpose(img, [1,0, 2])
img = np.reshape(img, [274, 7,14, 3])
print(target.shape)
np.save('data_no',img)
np.save('traget_no', target)
for x in range(274):
    plt.imshow(img[x])
    plt.show()


